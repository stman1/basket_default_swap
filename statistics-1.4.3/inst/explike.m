## Copyright (C) 2021 Nir Krakauer <nkrakauer@ccny.cuny.edu>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {@var{nlogL}, @var{avar} =} explike (@var{param}, @var{data})
## Compute the negative log-likelihood of data under the exponential distribution with given parameter value.
##
## @subheading Arguments
##
## @itemize @bullet
## @item
## @var{param} is a scalar containing the scale parameter of the exponential distribution (equal to its mean).
## @item
## @var{data} is the vector of given values.
##
## @end itemize
##
## @subheading Return values
##
## @itemize @bullet
## @item
## @var{nlogL} is the negative log-likelihood.
## @item
## @var{avar} is the inverse of the Fisher information matrix.
## (The Fisher information matrix is the second derivative of the negative log likelihood with respect to the parameter value.)
## 
## @end itemize
##
## @seealso{expcdf, expfit, expinv, explike, exppdf, exprnd}
## @end deftypefn

## Author: Nir Krakauer <nkrakauer@ccny.cuny.edu>
## Description: Negative log-likelihood for the exponential distribution

function [nlogL, avar] = explike (param, data)

  # Check arguments
  if (nargin != 2)
    print_usage;
  endif
  
  beta = param(1);
  n = numel (data);
  sx = sum (data(:));
  sxb = sx/beta;
  
  #calculate negative log likelihood
  nlogL = sxb + n*log(beta);
  
  #optionally calculate the inverse (reciprocal) of the second derivative of the negative log likelihood with respect to parameter
  if nargout > 1
    avar = (beta^2) ./ (2*sxb - n);
  endif

endfunction


%!test
%! x = 12;
%! beta = 5;
%! [L, V] = explike (beta, x);
%! expected_L = 4.0094;
%! expected_V = 6.5789;
%! assert (L, expected_L, 0.001);
%! assert (V, expected_V, 0.001);

%!test
%! x = 1:5;
%! beta = 2;
%! [L, V] = explike (beta, x);
%! expected_L = 10.9657;
%! expected_V = 0.4;
%! assert (L, expected_L, 0.001);
%! assert (V, expected_V, 0.001);

