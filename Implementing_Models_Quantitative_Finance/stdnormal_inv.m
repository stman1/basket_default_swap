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
## @deftypefn {} {} stdnormal_inv (@var{x})
## For each element of @var{x}, compute the quantile (the
## inverse of the CDF) at @var{x} of the standard normal distribution
## (mean = 0, standard deviation = 1).
## @end deftypefn

## Author: KH <Kurt.Hornik@wu-wien.ac.at>
## Description: Quantile function of the standard normal distribution

function inv = stdnormal_inv (x)

  if (nargin != 1)
    print_usage ();
  endif

  if (iscomplex (x))
    error ("stdnormal_inv: X must not be complex");
  endif

  inv = - sqrt (2) * erfcinv (2 * x);

endfunction


%!shared x
%! x = [-1 0 0.5 1 2];
%!assert (stdnormal_inv (x), [NaN -Inf 0 Inf NaN])

## Test class of input preserved
%!assert (stdnormal_inv ([x, NaN]), [NaN -Inf 0 Inf NaN NaN])
%!assert (stdnormal_inv (single ([x, NaN])), single ([NaN -Inf 0 Inf NaN NaN]))

## Test input validation
%!error stdnormal_inv ()
%!error stdnormal_inv (1,2)
%!error stdnormal_inv (i)
