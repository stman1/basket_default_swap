## Copyright (C) 2012 Rik Wehbring
## Copyright (C) 1995-2016 Kurt Hornik
##
## This program is free software: you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} binocdf (@var{x}, @var{n}, @var{p})
## @deftypefnx {Function File} {} binocdf (@var{x}, @var{n}, @var{p}, 'upper')
##
## For each element of @var{x}, compute the cumulative distribution function
## (CDF) at @var{x} of the binomial distribution with parameters @var{n} and
## @var{p}, where @var{n} is the number of trials and @var{p} is the
## probability of success.
##
## binocdf (@var{x}, @var{n}, @var{p}, 'upper') computes the complement
## of the cumulative distribution function.
## @end deftypefn

## Author: KH <Kurt.Hornik@wu-wien.ac.at>
## Description: CDF of the binomial distribution

function cdf = binocdf (x, n, p, upper)

  if (nargin != 3 && nargin != 4)
    print_usage ();
  endif

  if (nargin == 4)
    if (strcmp (upper, 'upper'))
       upper = true;
    else
       error ("binocdf: 4th parameter can only be  'upper'\n");
    endif
  else
     upper = false;
  endif

  if (! isscalar (n) || ! isscalar (p))
    [retval, x, n, p] = common_size (x, n, p);
    if (retval > 0)
      error ("binocdf: X, N, and P must be of common size or scalars");
    endif
  endif

  if (iscomplex (x) || iscomplex (n) || iscomplex (p))
    error ("binocdf: X, N, and P must not be complex");
  endif

  if (isa (x, "single") || isa (n, "single") || isa (p, "single"));
    cdf = nan (size (x), "single");
  else
    cdf = nan (size (x));
  endif

  k = (x >= n) & (n >= 0) & (n == fix (n) & (p >= 0) & (p <= 1));
  cdf(k) = !upper;

  k = (x < 0) & (n >= 0) & (n == fix (n) & (p >= 0) & (p <= 1));
  cdf(k) = upper;

  k = (x >= 0) & (x < n) & (n == fix (n)) & (p >= 0) & (p <= 1);
  tmp = floor (x(k));
  if !upper
    if (isscalar (n) && isscalar (p))
      cdf(k) = betainc (1 - p, n - tmp, tmp + 1);
    else
      cdf(k) = betainc (1 - p(k), n(k) - tmp, tmp + 1);
    endif
  else
    if (isscalar (n) && isscalar (p));
      cdf(k) = betainc (p, tmp + 1, n - tmp);
    else
      cdf(k) = betainc (p(k), tmp + 1, n(k) - tmp);
    endif
  endif

endfunction


%!shared x,y,y1
%! x = [-1 0 1 2 3];
%! y = [0 1/4 3/4 1 1];
%! y1 = 1-y;
%!assert (binocdf (x, 2*ones (1,5), 0.5*ones (1,5)), y, eps)
%!assert (binocdf (x, 2, 0.5*ones (1,5)), y, eps)
%!assert (binocdf (x, 2*ones (1,5), 0.5), y, eps)
%!assert (binocdf (x, 2*[0 -1 NaN 1.1 1], 0.5), [0 NaN NaN NaN 1])
%!assert (binocdf (x, 2, 0.5*[0 -1 NaN 3 1]), [0 NaN NaN NaN 1])
%!assert (binocdf ([x(1:2) NaN x(4:5)], 2, 0.5), [y(1:2) NaN y(4:5)], eps)
%!assert (binocdf(99, 100, 0.1, 'upper'), 1e-100, 1e-112);
%!assert (binocdf (x, 2*ones (1,5), 0.5*ones (1,5), 'upper'), y1, eps)
%!assert (binocdf (x, 2, 0.5*ones (1,5), 'upper'), y1, eps)
%!assert (binocdf (x, 2*ones (1,5), 0.5, 'upper'), y1, eps)
%!assert (binocdf (x, 2*[0 -1 NaN 1.1 1], 0.5, 'upper'), [1 NaN NaN NaN 0])
%!assert (binocdf (x, 2, 0.5*[0 -1 NaN 3 1], 'upper'), [1 NaN NaN NaN 0])
%!assert (binocdf ([x(1:2) NaN x(4:5)], 2, 0.5, 'upper'), [y1(1:2) NaN y1(4:5)])

## Test class of input preserved
%!assert (binocdf ([x, NaN], 2, 0.5), [y, NaN], eps)
%!assert (binocdf (single ([x, NaN]), 2, 0.5), single ([y, NaN]))
%!assert (binocdf ([x, NaN], single (2), 0.5), single ([y, NaN]))
%!assert (binocdf ([x, NaN], 2, single (0.5)), single ([y, NaN]))

## Test input validation
%!error binocdf ()
%!error binocdf (1)
%!error binocdf (1,2)
%!error binocdf (1,2,3,4)
%!error binocdf (ones (3), ones (2), ones (2))
%!error binocdf (ones (2), ones (3), ones (2))
%!error binocdf (ones (2), ones (2), ones (3))
%!error binocdf (i, 2, 2)
%!error binocdf (2, i, 2)
%!error binocdf (2, 2, i)
