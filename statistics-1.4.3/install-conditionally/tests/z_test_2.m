## Copyright (C) 1995-2017 Kurt Hornik
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
## @deftypefn {} {[@var{pval}, @var{z}] =} z_test_2 (@var{x}, @var{y}, @var{v_x}, @var{v_y}, @var{alt})
## For two samples @var{x} and @var{y} from normal distributions with unknown
## means and known variances @var{v_x} and @var{v_y}, perform a Z-test of the
## hypothesis of equal means.
##
## Under the null, the test statistic @var{z} follows a standard normal
## distribution.
##
## With the optional argument string @var{alt}, the alternative of interest
## can be selected.  If @var{alt} is @qcode{"!="} or @qcode{"<>"}, the null
## is tested against the two-sided alternative
## @code{mean (@var{x}) != mean (@var{y})}.  If alt is @qcode{">"}, the
## one-sided alternative @code{mean (@var{x}) > mean (@var{y})} is used.
## Similarly for @qcode{"<"}, the one-sided alternative
## @code{mean (@var{x}) < mean (@var{y})} is used.  The default is the
## two-sided case.
##
## The p-value of the test is returned in @var{pval}.
##
## If no output argument is given, the p-value of the test is displayed along
## with some information.
## @end deftypefn

## Author: KH <Kurt.Hornik@wu-wien.ac.at>
## Description: Compare means of two normal samples with known variances

function [pval, z] = z_test_2 (x, y, v_x, v_y, alt)

  if (nargin < 4 || nargin > 5)
    print_usage ();
  endif

  if (! (isvector (x) && isvector (y)))
    error ("z_test_2: both X and Y must be vectors");
  elseif (! (isscalar (v_x) && (v_x > 0)
             && isscalar (v_y) && (v_y > 0)))
    error ("z_test_2: both V_X and V_Y must be positive scalars");
  endif

  n_x  = length (x);
  n_y  = length (y);
  mu_x = sum (x) / n_x;
  mu_y = sum (y) / n_y;
  z    = (mu_x - mu_y) / sqrt (v_x / n_x + v_y / n_y);
  cdf  = stdnormal_cdf (z);

  if (nargin == 4)
    alt = "!=";
  endif

  if (! ischar (alt))
    error ("z_test_2: ALT must be a string");
  elseif (strcmp (alt, "!=") || strcmp (alt, "<>"))
    pval = 2 * min (cdf, 1 - cdf);
  elseif (strcmp (alt, ">"))
    pval = 1 - cdf;
  elseif (strcmp (alt, "<"))
    pval = cdf;
  else
    error ("z_test_2: option %s not recognized", alt);
  endif

  if (nargout == 0)
    s = ["Two-sample Z-test of mean(x) == mean(y) against ", ...
         "mean(x) %s mean(y),\n",                            ...
         "with known var(x) == %g and var(y) == %g:\n",      ...
         "  pval = %g\n"];
    printf (s, alt, v_x, v_y, pval);
  endif

endfunction

%!test
%! ## Two-sided (also the default option)
%! x = randn (100, 1); v_x = 2; x = v_x * x;
%! [pval, zval] = z_test_2 (x, x, v_x, v_x);
%! zval_exp = 0; pval_exp = 1.0;
%! assert (zval, zval_exp, eps);
%! assert (pval, pval_exp, eps);

%!test
%! ## Two-sided (also the default option)
%! x = randn (10000, 1); v_x = 2; x = v_x * x; n_x = length (x);
%! y = randn (20000, 1); v_y = 3; y = v_y * y; n_y = length (y);
%! [pval, z] = z_test_2 (x, y, v_x, v_y);
%! if (mean (x) >= mean (y))
%!   zval = abs (norminv (0.5*pval));
%! else
%!   zval = -abs (norminv (0.5*pval));
%! endif
%! unew = zval * sqrt (v_x/n_x + v_y/n_y);
%! delmu = mean (x) - mean (y);
%! assert (delmu, unew, 100*eps);

%!test
%! x = randn (100, 1); v_x = 2; x = v_x * x;
%! [pval, zval] = z_test_2 (x, x, v_x, v_x, ">");
%! zval_exp = 0; pval_exp = 0.5;
%! assert (zval, zval_exp, eps);
%! assert (pval, pval_exp, eps);

%!test
%! x = randn (10000, 1); v_x = 2; x = v_x * x; n_x = length (x);
%! y = randn (20000, 1); v_y = 3; y = v_y * y; n_y = length (y);
%! [pval, z] = z_test_2 (x, y, v_x, v_y, ">");
%! zval = norminv (1-pval);
%! unew = zval * sqrt (v_x/n_x + v_y/n_y);
%! delmu = mean (x) - mean (y);
%! assert (delmu, unew, 100*eps);

%!test
%! x = randn (100, 1); v_x = 2; x = v_x * x;
%! [pval, zval] = z_test_2 (x, x, v_x, v_x, "<");
%! zval_exp = 0; pval_exp = 0.5;
%! assert (zval, zval_exp, eps);
%! assert (pval, pval_exp, eps);

%!test
%! x = randn (10000, 1); v_x = 2; x = v_x * x; n_x = length (x);
%! y = randn (20000, 1); v_y = 3; y = v_y * y; n_y = length (y);
%! [pval, z] = z_test_2 (x, y, v_x, v_y, "<");
%! zval = norminv (pval);
%! unew = zval * sqrt (v_x/n_x + v_y/n_y);
%! delmu = mean (x) - mean (y);
%! assert (delmu, unew, 100*eps);
