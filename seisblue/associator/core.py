import datetime
from operator import itemgetter
import time
import tqdm
from sqlalchemy import Index

import scipy.signal
import matplotlib.pyplot as plt

import seisblue.associator.sql
from seisblue.associator.sql import AssociatedEvent, Candidate, PicksAssoc
from seisblue.associator.utils import time_correction, hypo_search


class LocalAssociator:
    """
    The Associator associate picks with travel time curve of velocity.
    """

    def __init__(self,
                 database,
                 max_s_p=60,
                 min_s_p=1.0,
                 origin_time_delta=3,
                 nsta_declare=3):
        """
        Parameters:
        assoc_db: associator database
        max_s_p: maximum time between s and p phase pick
        assoc_ot_uncert: origin time uncertainty window
        nsta_declare: minimum station number to declare a earthquake
        """
        self.database = database
        self.db = seisblue.associator.sql.Client(database, build=True)
        self.session = self.db.session
        self.max_s_p = max_s_p
        self.min_s_p = min_s_p
        self.origin_time_delta = datetime.timedelta(seconds=origin_time_delta)
        self.nsta_declare = nsta_declare

    def id_candidate_events(self):
        """
        Create a set of possible candidate events from our picks table.

        Where session is the connection to the sqlalchemy database.
        This method simply takes all picks with time differences
        less than our maximum S-P times for each station and generates
        a list of candidate events.
        """
        now1 = time.time()
        #############
        # Get all stations with unnassoiated picks
        stations = self.session.query(PicksAssoc.sta) \
            .filter(PicksAssoc.assoc_id == None) \
            .distinct() \
            .all()

        stations = seisblue.utils.flatten_list(stations)

        for sta in stations:

            nets = self.session.query(PicksAssoc.net) \
                .filter(PicksAssoc.sta == sta) \
                .filter(PicksAssoc.assoc_id == None) \
                .all()
            locs = self.session.query(PicksAssoc.loc) \
                .filter(PicksAssoc.sta == sta) \
                .filter(PicksAssoc.assoc_id == None) \
                .all()

            for net, in set(nets):
                for loc, in set(locs):

                    p_picks = self.session.query(PicksAssoc) \
                        .filter(PicksAssoc.sta == sta,
                                PicksAssoc.net == net,
                                PicksAssoc.loc == loc,
                                PicksAssoc.phase == 'P') \
                        .filter(PicksAssoc.assoc_id == None) \
                        .order_by(PicksAssoc.time) \
                        .all()
                    s_picks = self.session.query(PicksAssoc) \
                        .filter(PicksAssoc.sta == sta,
                                PicksAssoc.net == net,
                                PicksAssoc.loc == loc,
                                PicksAssoc.phase == 'S') \
                        .filter(PicksAssoc.assoc_id == None) \
                        .order_by(PicksAssoc.time) \
                        .all()

                    # Generate all possible candidate events
                    for p_pick in p_picks:
                        for s_pick in s_picks:
                            s_p = (s_pick.time - p_pick.time).total_seconds()
                            if s_p > self.max_s_p:
                                break
                            if s_p >= self.min_s_p:
                                origin_time = self.estimate_origin_time(p_pick,
                                                                        s_p)
                                new_candidate = Candidate(
                                    origin_time,
                                    sta,
                                    p_pick.time,
                                    p_pick.id,
                                    s_pick.time,
                                    s_pick.id
                                )
                                self.session.add(new_candidate)
            self.session.commit()
        Index('origin_time', Candidate.origin_time).create(self.db.engine)
        Index('P_time', Candidate.p_time).create(self.db.engine)
        Index('S_time', Candidate.s_time).create(self.db.engine)
        print(f'id_candidate time in seconds: {time.time() - now1}')

    def fast_associate(self, plot=False):
        now2 = time.time()

        candidates = self.session.query(Candidate) \
            .filter(Candidate.assoc_id == None) \
            .order_by(Candidate.origin_time) \
            .all()

        candidate_nsta = []
        for candidate in candidates:
            nsta = self.session.query(Candidate.sta) \
                .filter(Candidate.assoc_id == None) \
                .filter(Candidate.origin_time >= candidate.origin_time,
                        Candidate.origin_time < (candidate.origin_time +
                                                 self.origin_time_delta)) \
                .distinct() \
                .all()

            candidate_nsta.append(len(nsta))

        if plot:
            plt.plot(candidate_nsta)
            plt.hlines(3, 0, len(candidate_nsta), 'r', linestyles=':')
            plt.show()

        peaks, _ = scipy.signal.find_peaks(
            candidate_nsta,
            height=self.nsta_declare,
            distance=self.nsta_declare)

        peak_candidates = [[candidates[i], candidate_nsta[i]] for i in peaks]

        peak_candidates.sort(key=itemgetter(1), reverse=True)
        print(f'candidate_nsta time in seconds : {time.time() - now2}, ',
              f'candidate list length: {len(peak_candidates)}')

        print('Minimum station count for locating:', self.nsta_declare)
        for pc in tqdm.tqdm(peak_candidates):
            assoc_parallel(pc, database=self.database,
                           origin_time_delta=self.origin_time_delta,
                           nsta_declare=self.nsta_declare)

        assoc = self.session.query(AssociatedEvent).all()
        print(f'associate time in seconds : {time.time() - now2}, ',
              f'associated {len(assoc)} events')

    @staticmethod
    def remove_dupl_sta_candidate(match_candidates):
        stations = [candidate.sta for candidate in match_candidates]
        duplicate_stations = LocalAssociator.list_duplicates(stations)
        no_dupl_sta_candidates = [candidate for candidate in match_candidates
                                  if candidate.sta not in duplicate_stations]

        for sta in duplicate_stations:
            dupl_sta_candidates = [candidate for candidate in match_candidates
                                   if candidate.sta == sta]
            no_dupl_sta_candidates.append(dupl_sta_candidates[0])
        return no_dupl_sta_candidates

    @staticmethod
    def estimate_origin_time(p_arrival,
                             s_p_time,
                             vp_vs_ratio=1.75,
                             depth_correction=0.0):
        """
        Calculate approximate origin time by slope.

        y = ax + b, where y = p travel time, x = s_p time
        a = 1/(vp_vs_ratio - 1), b = depth correction

        Vp/Vs in theory is sqrt(3) or 1.73, in practice is near around 1.78.
        Depth correction will not take any effect on candidate clustering
        because all picks will shift in same amount of time.
        """
        p_travel_time = s_p_time / (vp_vs_ratio - 1) + depth_correction
        origin_time = p_arrival.time - datetime.timedelta(
            seconds=p_travel_time)

        return origin_time

    @staticmethod
    def list_duplicates(seq):
        seen = set()
        seen_add = seen.add
        seen_twice = set(x for x in seq if x in seen or seen_add(x))
        return list(seen_twice)


def assoc_parallel(peak_candidates, database, origin_time_delta, nsta_declare):
    db = seisblue.associator.sql.Client(database)
    candidate = peak_candidates[0]
    nsta = peak_candidates[1]
    print(
        f'time: {str(peak_candidates[0].origin_time)}, station count: {nsta}')
    match_candidates = db.session.query(Candidate) \
        .filter(Candidate.assoc_id == None) \
        .filter(Candidate.origin_time >= candidate.origin_time) \
        .filter(Candidate.origin_time < (candidate.origin_time +
                                         origin_time_delta * 2)) \
        .order_by(Candidate.origin_time) \
        .all()

    if len(match_candidates) < nsta_declare:
        print('no enough picks left')
        return

    # remove the candidates with the picks has been associated
    associated_pick_id = db.session.query(PicksAssoc.id) \
        .filter(PicksAssoc.assoc_id != None) \
        .distinct() \
        .all()

    associated_pick_id = seisblue.utils.flatten_list(
        associated_pick_id)

    match_candidates = [candidate for candidate in match_candidates
                        if candidate.p_id not in associated_pick_id
                        and candidate.s_id not in associated_pick_id]

    if len(match_candidates) < nsta_declare:
        print('no enough picks left by remove associated picks')
        return

    match_candidates = LocalAssociator.remove_dupl_sta_candidate(
        match_candidates)

    if len(match_candidates) < nsta_declare:
        print('no enough picks left by remove multi pick station')
        return

    new_match_candidates, origin, QA = hypo_search(match_candidates)
    print(f'Hypocenter searching done, result: {QA}')

    if not QA:
        return

    if len(new_match_candidates) < nsta_declare:
        print('no enough picks left by hypo search')
        return

    nsta = len(new_match_candidates)
    new_event = AssociatedEvent(origin.time.datetime,
                                0,
                                origin.latitude,
                                origin.longitude,
                                origin.depth,
                                nsta,
                                origin.time_errors.uncertainty,
                                origin.longitude_errors.uncertainty,
                                origin.latitude_errors.uncertainty,
                                origin.depth_errors.uncertainty,
                                origin.quality.azimuthal_gap
                                )
    db.session.add(new_event)
    db.session.flush()
    db.session.refresh(new_event)

    for match_candidate in new_match_candidates:
        match_candidate.set_assoc_id(new_event.id, db.session, True)
    db.session.commit()

    return


if __name__ == "__main__":
    pass
