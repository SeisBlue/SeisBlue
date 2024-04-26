# -*- coding: utf-8 -*-
import node03_test as test
import node04_to_sfile as to_sfile
import node05_SEISAN_FPFIT_AUTOMAG as fp_mag
import node06_Kagan_test as kagan_test


def main():
    test.main()
    to_sfile.main()
    fp_mag.main()
    # kagan_test.main()


if __name__ == '__main__':
    main()