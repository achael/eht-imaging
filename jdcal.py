# -*- coding:utf-8 -*-
"""Functions for converting between Julian dates and calendar dates.

A function for converting Gregorian calendar dates to Julian dates, and
another function for converting Julian calendar dates to Julian dates
are defined. Two functions for the reverse calculations are also
defined.

Different regions of the world switched to Gregorian calendar from
Julian calendar on different dates. Having separate functions for Julian
and Gregorian calendars allow maximum flexibility in choosing the
relevant calendar.

All the above functions are "proleptic". This means that they work for
dates on which the concerned calendar is not valid. For example,
Gregorian calendar was not used prior to around October 1582.

Julian dates are stored in two floating point numbers (double).  Julian
dates, and Modified Julian dates, are large numbers. If only one number
is used, then the precision of the time stored is limited. Using two
numbers, time can be split in a manner that will allow maximum
precision. For example, the first number could be the Julian date for
the beginning of a day and the second number could be the fractional
day. Calculations that need the latter part can now work with maximum
precision.

A function to test if a given Gregorian calendar year is a leap year is
defined.

Zero point of Modified Julian Date (MJD) and the MJD of 2000/1/1
12:00:00 are also given.

This module is based on the TPM C library, by Jeffery W. Percival. The
idea for splitting Julian date into two floating point numbers was
inspired by the IAU SOFA C library.

:author: Prasanth Nair
:contact: prasanthhn@gmail.com
:license: BSD (http://www.opensource.org/licenses/bsd-license.php)
"""
from __future__ import division
from __future__ import print_function
import math

__version__ = "1.0.1"

MJD_0 = 2400000.5
MJD_JD2000 = 51544.5


def fpart(x):
    """Return fractional part of given number."""
    return math.modf(x)[0]


def ipart(x):
    """Return integer part of given number."""
    return math.modf(x)[1]


def is_leap(year):
    """Leap year or not in the Gregorian calendar."""
    x = math.fmod(year, 4)
    y = math.fmod(year, 100)
    z = math.fmod(year, 400)

    # Divisible by 4 and,
    # either not divisible by 100 or divisible by 400.
    return not x and (y or not z)


def gcal2jd(year, month, day):
    """Gregorian calendar date to Julian date.

    The input and output are for the proleptic Gregorian calendar,
    i.e., no consideration of historical usage of the calendar is
    made.

    Parameters
    ----------
    year : int
        Year as an integer.
    month : int
        Month as an integer.
    day : int
        Day as an integer.

    Returns
    -------
    jd1, jd2: 2-element tuple of floats
        When added together, the numbers give the Julian date for the
        given Gregorian calendar date. The first number is always
        MJD_0 i.e., 2451545.5. So the second is the MJD.

    Examples
    --------
    >>> gcal2jd(2000,1,1)
    (2400000.5, 51544.0)
    >>> 2400000.5 + 51544.0 + 0.5
    2451545.0
    >>> year = [-4699, -2114, -1050, -123, -1, 0, 1, 123, 1678.0, 2000,
    ....: 2012, 2245]
    >>> month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> day = [1, 12, 23, 14, 25, 16, 27, 8, 9, 10, 11, 31]
    >>> x = [gcal2jd(y, m, d) for y, m, d in zip(year, month, day)]
    >>> for i in x: print i
    (2400000.5, -2395215.0)
    (2400000.5, -1451021.0)
    (2400000.5, -1062364.0)
    (2400000.5, -723762.0)
    (2400000.5, -679162.0)
    (2400000.5, -678774.0)
    (2400000.5, -678368.0)
    (2400000.5, -633797.0)
    (2400000.5, -65812.0)
    (2400000.5, 51827.0)
    (2400000.5, 56242.0)
    (2400000.5, 141393.0)

    Negative months and days are valid. For example, 2000/-2/-4 =>
    1999/+12-2/-4 => 1999/10/-4 => 1999/9/30-4 => 1999/9/26.

    >>> gcal2jd(2000, -2, -4)
    (2400000.5, 51447.0)
    >>> gcal2jd(1999, 9, 26)
    (2400000.5, 51447.0)

    >>> gcal2jd(2000, 2, -1)
    (2400000.5, 51573.0)
    >>> gcal2jd(2000, 1, 30)
    (2400000.5, 51573.0)

    >>> gcal2jd(2000, 3, -1)
    (2400000.5, 51602.0)
    >>> gcal2jd(2000, 2, 28)
    (2400000.5, 51602.0)

    Month 0 becomes previous month.

    >>> gcal2jd(2000, 0, 1)
    (2400000.5, 51513.0)
    >>> gcal2jd(1999, 12, 1)
    (2400000.5, 51513.0)

    Day number 0 becomes last day of previous month.

    >>> gcal2jd(2000, 3, 0)
    (2400000.5, 51603.0)
    >>> gcal2jd(2000, 2, 29)
    (2400000.5, 51603.0)

    If `day` is greater than the number of days in `month`, then it
    gets carried over to the next month.

    >>> gcal2jd(2000,2,30)
    (2400000.5, 51604.0)
    >>> gcal2jd(2000,3,1)
    (2400000.5, 51604.0)

    >>> gcal2jd(2001,2,30)
    (2400000.5, 51970.0)
    >>> gcal2jd(2001,3,2)
    (2400000.5, 51970.0)

    Notes
    -----
    The returned Julian date is for mid-night of the given date. To
    find the Julian date for any time of the day, simply add time as a
    fraction of a day. For example Julian date for mid-day can be
    obtained by adding 0.5 to either the first part or the second
    part. The latter is preferable, since it will give the MJD for the
    date and time.

    BC dates should be given as -(BC - 1) where BC is the year. For
    example 1 BC == 0, 2 BC == -1, and so on.

    Negative numbers can be used for `month` and `day`. For example
    2000, -1, 1 is the same as 1999, 11, 1.

    The Julian dates are proleptic Julian dates, i.e., values are
    returned without considering if Gregorian dates are valid for the
    given date.

    The input values are truncated to integers.

    """
    year = int(year)
    month = int(month)
    day = int(day)

    a = ipart((month - 14) / 12.0)
    jd = ipart((1461 * (year + 4800 + a)) / 4.0)
    jd += ipart((367 * (month - 2 - 12 * a)) / 12.0)
    x = ipart((year + 4900 + a) / 100.0)
    jd -= ipart((3 * x) / 4.0)
    jd += day - 2432075.5  # was 32075; add 2400000.5

    jd -= 0.5  # 0 hours; above JD is for midday, switch to midnight.

    return MJD_0, jd


def jd2gcal(jd1, jd2):
    """Julian date to Gregorian calendar date and time of day.

    The input and output are for the proleptic Gregorian calendar,
    i.e., no consideration of historical usage of the calendar is
    made.

    Parameters
    ----------
    jd1, jd2: int
        Sum of the two numbers is taken as the given Julian date. For
        example `jd1` can be the zero point of MJD (MJD_0) and `jd2`
        can be the MJD of the date and time. But any combination will
        work.

    Returns
    -------
    y, m, d, f : int, int, int, float
        Four element tuple containing year, month, day and the
        fractional part of the day in the Gregorian calendar. The first
        three are integers, and the last part is a float.

    Examples
    --------
    >>> jd2gcal(*gcal2jd(2000,1,1))
    (2000, 1, 1, 0.0)
    >>> jd2gcal(*gcal2jd(1950,1,1))
    (1950, 1, 1, 0.0)

    Out of range months and days are carried over to the next/previous
    year or next/previous month. See gcal2jd for more examples.

    >>> jd2gcal(*gcal2jd(1999,10,12))
    (1999, 10, 12, 0.0)
    >>> jd2gcal(*gcal2jd(2000,2,30))
    (2000, 3, 1, 0.0)
    >>> jd2gcal(*gcal2jd(-1999,10,12))
    (-1999, 10, 12, 0.0)
    >>> jd2gcal(*gcal2jd(2000, -2, -4))
    (1999, 9, 26, 0.0)

    >>> gcal2jd(2000,1,1)
    (2400000.5, 51544.0)
    >>> jd2gcal(2400000.5, 51544.0)
    (2000, 1, 1, 0.0)
    >>> jd2gcal(2400000.5, 51544.5)
    (2000, 1, 1, 0.5)
    >>> jd2gcal(2400000.5, 51544.245)
    (2000, 1, 1, 0.24500000000261934)
    >>> jd2gcal(2400000.5, 51544.1)
    (2000, 1, 1, 0.099999999998544808)
    >>> jd2gcal(2400000.5, 51544.75)
    (2000, 1, 1, 0.75)

    Notes
    -----
    The last element of the tuple is the same as

       (hh + mm / 60.0 + ss / 3600.0) / 24.0

    where hh, mm, and ss are the hour, minute and second of the day.

    See Also
    --------
    gcal2jd

    """
    from math import modf

    jd1_f, jd1_i = modf(jd1)
    jd2_f, jd2_i = modf(jd2)

    jd_i = jd1_i + jd2_i

    f = jd1_f + jd2_f

    # Set JD to noon of the current date. Fractional part is the
    # fraction from midnight of the current date.
    if -0.5 < f < 0.5:
        f += 0.5
    elif f >= 0.5:
        jd_i += 1
        f -= 0.5
    elif f <= -0.5:
        jd_i -= 1
        f += 1.5

    l = jd_i + 68569
    n = ipart((4 * l) / 146097.0)
    l -= ipart(((146097 * n) + 3) / 4.0)
    i = ipart((4000 * (l + 1)) / 1461001)
    l -= ipart((1461 * i) / 4.0) - 31
    j = ipart((80 * l) / 2447.0)
    day = l - ipart((2447 * j) / 80.0)
    l = ipart(j / 11.0)
    month = j + 2 - (12 * l)
    year = 100 * (n - 49) + i + l

    return int(year), int(month), int(day), f


def jcal2jd(year, month, day):
    """Julian calendar date to Julian date.

    The input and output are for the proleptic Julian calendar,
    i.e., no consideration of historical usage of the calendar is
    made.

    Parameters
    ----------
    year : int
        Year as an integer.
    month : int
        Month as an integer.
    day : int
        Day as an integer.

    Returns
    -------
    jd1, jd2: 2-element tuple of floats
        When added together, the numbers give the Julian date for the
        given Julian calendar date. The first number is always
        MJD_0 i.e., 2451545.5. So the second is the MJD.

    Examples
    --------
    >>> jcal2jd(2000, 1, 1)
    (2400000.5, 51557.0)
    >>> year = [-4699, -2114, -1050, -123, -1, 0, 1, 123, 1678, 2000,
       ...:  2012, 2245]
    >>> month = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    >>> day = [1, 12, 23, 14, 25, 16, 27, 8, 9, 10, 11, 31]
    >>> x = [jcal2jd(y, m, d) for y, m, d in zip(year, month, day)]
    >>> for i in x: print i
    (2400000.5, -2395252.0)
    (2400000.5, -1451039.0)
    (2400000.5, -1062374.0)
    (2400000.5, -723765.0)
    (2400000.5, -679164.0)
    (2400000.5, -678776.0)
    (2400000.5, -678370.0)
    (2400000.5, -633798.0)
    (2400000.5, -65772.0)
    (2400000.5, 51871.0)
    (2400000.5, 56285.0)

    Notes
    -----
    Unlike `gcal2jd`, negative months and days can result in incorrect
    Julian dates.

    """
    year = int(year)
    month = int(month)
    day = int(day)

    jd = 367 * year
    x = ipart((month - 9) / 7.0)
    jd -= ipart((7 * (year + 5001 + x)) / 4.0)
    jd += ipart((275 * month) / 9.0)
    jd += day
    jd += 1729777 - 2400000.5  # Return 240000.5 as first part of JD.

    jd -= 0.5  # Convert midday to midnight.

    return MJD_0, jd


def jd2jcal(jd1, jd2):
    """Julian calendar date for the given Julian date.

    The input and output are for the proleptic Julian calendar,
    i.e., no consideration of historical usage of the calendar is
    made.

    Parameters
    ----------
    jd1, jd2: int
        Sum of the two numbers is taken as the given Julian date. For
        example `jd1` can be the zero point of MJD (MJD_0) and `jd2`
        can be the MJD of the date and time. But any combination will
        work.

    Returns
    -------
    y, m, d, f : int, int, int, float
        Four element tuple containing year, month, day and the
        fractional part of the day in the Julian calendar. The first
        three are integers, and the last part is a float.

    Examples
    --------
    >>> jd2jcal(*jcal2jd(2000, 1, 1))
    (2000, 1, 1, 0.0)
    >>> jd2jcal(*jcal2jd(-4000, 10, 11))
    (-4000, 10, 11, 0.0)

    >>> jcal2jd(2000, 1, 1)
    (2400000.5, 51557.0)
    >>> jd2jcal(2400000.5, 51557.0)
    (2000, 1, 1, 0.0)
    >>> jd2jcal(2400000.5, 51557.5)
    (2000, 1, 1, 0.5)
    >>> jd2jcal(2400000.5, 51557.245)
    (2000, 1, 1, 0.24500000000261934)
    >>> jd2jcal(2400000.5, 51557.1)
    (2000, 1, 1, 0.099999999998544808)
    >>> jd2jcal(2400000.5, 51557.75)
    (2000, 1, 1, 0.75)

    """
    from math import modf

    jd1_f, jd1_i = modf(jd1)
    jd2_f, jd2_i = modf(jd2)

    jd_i = jd1_i + jd2_i

    f = jd1_f + jd2_f

    # Set JD to noon of the current date. Fractional part is the
    # fraction from midnight of the current date.
    if -0.5 < f < 0.5:
        f += 0.5
    elif f >= 0.5:
        jd_i += 1
        f -= 0.5
    elif f <= -0.5:
        jd_i -= 1
        f += 1.5

    j = jd_i + 1402.0
    k = ipart((j - 1) / 1461.0)
    l = j - (1461.0 * k)
    n = ipart((l - 1) / 365.0) - ipart(l / 1461.0)
    i = l - (365.0 * n) + 30.0
    j = ipart((80.0 * i) / 2447.0)
    day = i - ipart((2447.0 * j) / 80.0)
    i = ipart(j / 11.0)
    month = j + 2 - (12.0 * i)
    year = (4 * k) + n + i - 4716.0

    return int(year), int(month), int(day), f


# Some tests.
def _test_gcal2jd_with_sla_cldj():
    """Compare gcal2jd with slalib.sla_cldj."""
    import random
    try:
        from pyslalib import slalib
    except ImportError:
        print("SLALIB (PySLALIB not available).")
        return 1
    n = 1000
    mday = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # sla_cldj needs year > -4699 i.e., 4700 BC.
    year = [random.randint(-4699, 2200) for i in range(n)]
    month = [random.randint(1, 12) for i in range(n)]
    day = [random.randint(1, 31) for i in range(n)]
    for i in range(n):
        x = 0
        if is_leap(year[i]) and month[i] == 2:
            x = 1
        if day[i] > mday[month[i]] + x:
            day[i] = mday[month[i]]

    jd_jdc = [gcal2jd(y, m, d)[1]
              for y, m, d in zip(year, month, day)]
    jd_sla = [slalib.sla_cldj(y, m, d)[0]
              for y, m, d in zip(year, month, day)]
    diff = [abs(i - j) for i, j in zip(jd_sla, jd_jdc)]
    assert max(diff) <= 1e-8
    assert min(diff) <= 1e-8


def _test_jd2gcal():
    """Check jd2gcal as reverse of gcal2jd."""
    import random
    n = 1000
    mday = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    year = [random.randint(-4699, 2200) for i in range(n)]
    month = [random.randint(1, 12) for i in range(n)]
    day = [random.randint(1, 31) for i in range(n)]
    for i in range(n):
        x = 0
        if is_leap(year[i]) and month[i] == 2:
            x = 1
        if day[i] > mday[month[i]] + x:
            day[i] = mday[month[i]]

    jd = [gcal2jd(y, m, d)[1]
          for y, m, d in zip(year, month, day)]

    x = [jd2gcal(MJD_0, i) for i in jd]

    for i in range(n):
        assert x[i][0] == year[i]
        assert x[i][1] == month[i]
        assert x[i][2] == day[i]
        assert x[i][3] <= 1e-15


def _test_jd2jcal():
    """Check jd2jcal as reverse of jcal2jd."""
    import random
    n = 1000
    year = [random.randint(-4699, 2200) for i in range(n)]
    month = [random.randint(1, 12) for i in range(n)]
    day = [random.randint(1, 28) for i in range(n)]

    jd = [jcal2jd(y, m, d)[1]
          for y, m, d in zip(year, month, day)]

    x = [jd2gcal(MJD_0, i) for i in jd]

    for i in range(n):
        assert x[i][0] == year[i]
        assert x[i][1] == month[i]
        assert x[i][2] == day[i]
        assert x[i][3] <= 1e-15
