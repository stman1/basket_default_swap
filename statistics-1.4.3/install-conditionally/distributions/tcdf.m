## Copyright (C) 2013-2017 Julien Bect
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
## @deftypefn {} {} tcdf (@var{x}, @var{n})
## For each element of @var{x}, compute the cumulative distribution function
## (CDF) at @var{x} of the t (Student) distribution with
## @var{n} degrees of freedom.
## @end deftypefn

## Author: KH <Kurt.Hornik@wu-wien.ac.at>
## Description: CDF of the t distribution

function cdf = tcdf (x, n)

  if (nargin != 2)
    print_usage ();
  endif

  if (! isscalar (n))
    [retval, x, n] = common_size (x, n);
    if (retval > 0)
      error ("tcdf: X and N must be of common size or scalars");
    endif
  endif

  if (iscomplex (x) || iscomplex (n))
    error ("tcdf: X and N must not be complex");
  endif

  if (isa (x, "single") || isa (n, "single"))
    cdf = zeros (size (x), "single");
  else
    cdf = zeros (size (x));
  endif

  k = isnan (x) | !(n > 0);
  cdf(k) = NaN;

  k = (x == Inf) & (n > 0);
  cdf(k) = 1;
  #(for x == -Inf, the cdf is already 0, so doesn't need setting)

  k = isfinite (x) & (n > 0);

  max_int_n = 1E4;
  ks = k & (fix (n) == n) & (n <= max_int_n);
  if any (ks(:))
    if (isscalar (n))
      cdf(ks) = tcdf_integer_df (x(ks), n);
      return
    else
      nn = unique (n(ks));
      ni = numel (nn);
      for i = 1:ni
        ki = k & (n == nn(i));
        cdf(ki) = tcdf_integer_df (x(ki), nn(i));
      endfor
    endif
  endif

  k &= !ks;

  xx = x .^ 2;
  x_big_abs = (xx > n);

  ## deal with the case "abs(x) big"
  kk = k & x_big_abs;
  if (isscalar (n))
    cdf(kk) = betainc (n ./ (n + xx(kk)), n/2, 1/2) / 2;
  else
    cdf(kk) = betainc (n(kk) ./ (n(kk) + xx(kk)), n(kk)/2, 1/2) / 2;
  endif

  ## deal with the case "abs(x) small"
  kk = k & ! x_big_abs;
  if (isscalar (n))
    cdf(kk) = 0.5 * (1 - betainc (xx(kk) ./ (n + xx(kk)), 1/2, n/2));
  else
    cdf(kk) = 0.5 * (1 - betainc (xx(kk) ./ (n(kk) + xx(kk)), 1/2, n(kk)/2));
  endif

  k &= (x > 0);
  if (any (k(:)))
    cdf(k) = 1 - cdf(k);
  endif

endfunction


function cdf = tcdf_integer_df (x, n)

# compute the t distribution CDF efficiently (without calling betainc) when n is a small positive integer
# Reference: Christian Walck (2007), Hand-book on Statistical Distributions for Experimentalists, University of Stockholm Internal Report SUF-PFY/96-01, Section 38.12, url: http://www.fysik.su.se/~walck/suf9601.pdf

  if (n == 1)
    cdf = 0.5 + atan(x)/pi;
  
  elseif (n == 2)
    cdf = 0.5 + x ./ (2 * sqrt(2 + x .^ 2));
    
  else
    xs = x ./ sqrt(n);
    xxf = 1 ./ (1 + xs .^ 2);
    u = s = 1; 

    if mod (n, 2) #n odd

      m = (n - 1) / 2;      
      for i = 2:m
        u .*= (1 - 1/(2*i - 1)) .* xxf;
        s += u;
      endfor
      cdf = 0.5 + (xs .* xxf .* s + atan(xs)) / pi;      
      
    else #n even

      m = n / 2;
      for i = 1:(m - 1)
        u .*= (1 - 1/(2*i)) .* xxf;
        s += u;
      endfor
      cdf = 0.5 + (xs .* sqrt(xxf) .* s) / 2;

    endif

  endif
endfunction

%!shared x,y
%! x = [-Inf 0 1 Inf];
%! y = [0 1/2 3/4 1];
%!assert (tcdf (x, ones (1,4)), y, eps)
%!assert (tcdf (x, 1), y, eps)
%!assert (tcdf (x, [0 1 NaN 1]), [NaN 1/2 NaN 1], eps)
%!assert (tcdf ([x(1:2) NaN x(4)], 1), [y(1:2) NaN y(4)], eps)

## Test class of input preserved
%!assert (tcdf ([x, NaN], 1), [y, NaN], eps)
%!assert (tcdf (single ([x, NaN]), 1), single ([y, NaN]), eps ("single"))
%!assert (tcdf ([x, NaN], single (1)), single ([y, NaN]), eps ("single"))

## Test input validation
%!error tcdf ()
%!error tcdf (1)
%!error tcdf (1,2,3)
%!error tcdf (ones (3), ones (2))
%!error tcdf (ones (2), ones (3))
%!error tcdf (i, 2)
%!error tcdf (2, i)

## Check some reference values

%!shared tol_rel
%! tol_rel = 10 * eps;

## check accuracy for small positive values
%!assert (tcdf (10^(-10), 2.5), 0.50000000003618087, -tol_rel)
%!assert (tcdf (10^(-11), 2.5), 0.50000000000361809, -tol_rel)
%!assert (tcdf (10^(-12), 2.5), 0.50000000000036181, -tol_rel)
%!assert (tcdf (10^(-13), 2.5), 0.50000000000003618, -tol_rel)
%!assert (tcdf (10^(-14), 2.5), 0.50000000000000362, -tol_rel)
%!assert (tcdf (10^(-15), 2.5), 0.50000000000000036, -tol_rel)
%!assert (tcdf (10^(-16), 2.5), 0.50000000000000004, -tol_rel)

## check accuracy for large negative values
%!assert (tcdf (-10^1, 2.5), 2.2207478836537124e-03, -tol_rel)
%!assert (tcdf (-10^2, 2.5), 7.1916492116661878e-06, -tol_rel)
%!assert (tcdf (-10^3, 2.5), 2.2747463948307452e-08, -tol_rel)
%!assert (tcdf (-10^4, 2.5), 7.1933970159922115e-11, -tol_rel)
%!assert (tcdf (-10^5, 2.5), 2.2747519231756221e-13, -tol_rel)

## # Reference values obtained using Python 2.7.4 and mpmath 0.17
##
## from mpmath import *
##
## mp.dps = 100
##
## def F(x_in, nu_in):
##     x = mpf(x_in);
##     nu = mpf(nu_in);
##     t = nu / (nu + x*x)
##     a = nu / 2
##     b = mpf(0.5)
##     F = betainc(a, b, 0, t, regularized=True) / 2
##     if (x > 0):
##         F = 1 - F
##     return F
##
## nu = 2.5
##
## for i in range(1, 6):
##     x = - power(mpf(10), mpf(i))
##     print "%%!assert (tcdf (-10^%d, 2.5), %s, -eps)" \
##         % (i, nstr(F(x, nu), 17))
##
## for i in range(10, 17):
##     x = power(mpf(10), -mpf(i))
##     print "%%!assert (tcdf (10^(-%d), 2.5), %s, -eps)" \
##         % (i, nstr(F(x, nu), 17))
