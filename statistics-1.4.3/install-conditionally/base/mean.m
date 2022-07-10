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
## @deftypefn  {} {} mean (@var{x})
## @deftypefnx {} {} mean (@var{x}, @var{dim})
## @deftypefnx {} {} mean (@var{x}, @var{opt})
## @deftypefnx {} {} mean (@var{x}, @var{dim}, @var{opt})
## Compute the mean of the elements of the vector @var{x}.
##
## The mean is defined as
##
## @tex
## $$ {\rm mean}(x) = \bar{x} = {1\over N} \sum_{i=1}^N x_i $$
## where $N$ is the number of elements of @var{x}.
##
## @end tex
## @ifnottex
##
## @example
## mean (@var{x}) = SUM_i @var{x}(i) / N
## @end example
##
## where @math{N} is the length of the @var{x} vector.
##
## @end ifnottex
## If @var{x} is a matrix, compute the mean for each column and return them
## in a row vector.
##
## If the optional argument @var{dim} is given, operate along this dimension.
##
## The optional argument @var{opt} selects the type of mean to compute.
## The following options are recognized:
##
## @table @asis
## @item @qcode{"a"}
## Compute the (ordinary) arithmetic mean.  [default]
##
## @item @qcode{"g"}
## Compute the geometric mean.
##
## @item @qcode{"h"}
## Compute the harmonic mean.
## @end table
##
## Both @var{dim} and @var{opt} are optional.  If both are supplied, either
## may appear first.
## @seealso{median, mode}
## @end deftypefn

## Author: KH <Kurt.Hornik@wu-wien.ac.at>
## Description: Compute arithmetic, geometric, and harmonic mean

function retval = mean (x, opt1, opt2)

  if (nargin < 1 || nargin > 3)
    print_usage ();
  endif

  if (! (isnumeric (x) || islogical (x)))
    error ("mean: X must be a numeric vector or matrix");
  endif

  need_dim = false;

  if (nargin == 1)
    opt = "a";
    need_dim = true;
  elseif (nargin == 2)
    if (ischar (opt1))
      opt = opt1;
      need_dim = true;
    else
      dim = opt1;
      opt = "a";
    endif
  elseif (nargin == 3)
    if (ischar (opt1))
      opt = opt1;
      dim = opt2;
    elseif (ischar (opt2))
      opt = opt2;
      dim = opt1;
    else
      error ("mean: OPT must be a string");
    endif
  else
    print_usage ();
  endif

  nd = ndims (x);
  sz = size (x);
  if (need_dim)
    ## Find the first non-singleton dimension.
    (dim = find (sz > 1, 1)) || (dim = 1);
  else
    if (! (isscalar (dim) && dim == fix (dim) && dim > 0))
      error ("mean: DIM must be an integer and a valid dimension");
    endif
  endif

  n = size (x, dim);
  
  if (isempty (x))
    %% codepath for Matlab compatibility. empty x produces NaN output, but 
    %% for ndim > 2, output depends on size of x and whether DIM is set.  
    if ((nd == 2) && (max (sz) < 2) && need_dim)
      retval = NaN;
    else
      if (~need_dim)
        sz(dim) = 1;
      else  
        sz (find ((sz ~= 1), 1)) = 1;
      endif  
      retval = NaN (sz);
    endif

    if (isa (x, "single"))
      retval = single (retval);  
    endif
 
  elseif (strcmp (opt, "a"))
    retval = sum (x, dim) / n;

  elseif (strcmp (opt, "g"))
    if (all (x(:) >= 0))
      retval = exp (sum (log (x), dim) ./ n);
    else
      error ("mean: X must not contain any negative values");
    endif
  elseif (strcmp (opt, "h"))
    retval = n ./ sum (1 ./ x, dim);
  else
    error ("mean: option '%s' not recognized", opt);
  endif

endfunction


%!test
%! x = -10:10;
%! y = x';
%! z = [y, y+10];
%! assert (mean (x), 0);
%! assert (mean (y), 0);
%! assert (mean (z), [0, 10]);

## Test small numbers
%!assert (mean (repmat (0.1, 1, 1000), "g"), 0.1, 20*eps)

%!assert (mean (magic (3), 1), [5, 5, 5])
%!assert (mean (magic (3), 2), [5; 5; 5])
%!assert (mean ([2 8], "g"), 4)
%!assert (mean ([4 4 2], "h"), 3)
%!assert (mean (logical ([1 0 1 1])), 0.75)
%!assert (mean (single ([1 0 1 1])), single (0.75))
%!assert (mean ([1 2], 3), [1 2])

##tests for empty input Matlab compatibility (bug #48690)
%!assert (mean ([]), NaN)
%!assert (mean (single([])), single(NaN))
%!assert (mean (ones (0, 0, 0, 0)), NaN (1, 0, 0, 0))
%!assert (mean (ones (0, 0, 0, 1)), NaN (1, 0, 0, 1))
%!assert (mean (ones (0, 0, 0, 2)), NaN (1, 0, 0, 2))
%!assert (mean (ones (0, 0, 1, 0)), NaN (1, 0, 1, 0))
%!assert (mean (ones (0, 0, 1, 1)), NaN (1, 1, 1, 1))
%!assert (mean (ones (0, 0, 1, 2)), NaN (1, 0, 1, 2))
%!assert (mean (ones (0, 0, 2, 0)), NaN (1, 0, 2, 0))
%!assert (mean (ones (0, 0, 2, 1)), NaN (1, 0, 2, 1))
%!assert (mean (ones (0, 0, 2, 2)), NaN (1, 0, 2, 2))
%!assert (mean (ones (0, 1, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mean (ones (0, 1, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mean (ones (0, 1, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mean (ones (0, 1, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mean (ones (0, 1, 1, 1)), NaN (1, 1, 1, 1))
%!assert (mean (ones (0, 1, 1, 2)), NaN (1, 1, 1, 2))
%!assert (mean (ones (0, 1, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mean (ones (0, 1, 2, 1)), NaN (1, 1, 2, 1))
%!assert (mean (ones (0, 1, 2, 2)), NaN (1, 1, 2, 2))
%!assert (mean (ones (0, 2, 0, 0)), NaN (1, 2, 0, 0))
%!assert (mean (ones (0, 2, 0, 1)), NaN (1, 2, 0, 1))
%!assert (mean (ones (0, 2, 0, 2)), NaN (1, 2, 0, 2))
%!assert (mean (ones (0, 2, 1, 0)), NaN (1, 2, 1, 0))
%!assert (mean (ones (0, 2, 1, 1)), NaN (1, 2, 1, 1))
%!assert (mean (ones (0, 2, 1, 2)), NaN (1, 2, 1, 2))
%!assert (mean (ones (0, 2, 2, 0)), NaN (1, 2, 2, 0))
%!assert (mean (ones (0, 2, 2, 1)), NaN (1, 2, 2, 1))
%!assert (mean (ones (0, 2, 2, 2)), NaN (1, 2, 2, 2))
%!assert (mean (ones (1, 0, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mean (ones (1, 0, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mean (ones (1, 0, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mean (ones (1, 0, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mean (ones (1, 0, 1, 1)), NaN (1, 1, 1, 1))
%!assert (mean (ones (1, 0, 1, 2)), NaN (1, 1, 1, 2))
%!assert (mean (ones (1, 0, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mean (ones (1, 0, 2, 1)), NaN (1, 1, 2, 1))
%!assert (mean (ones (1, 0, 2, 2)), NaN (1, 1, 2, 2))
%!assert (mean (ones (1, 1, 0, 0)), NaN (1, 1, 1, 0))
%!assert (mean (ones (1, 1, 0, 1)), NaN (1, 1, 1, 1))
%!assert (mean (ones (1, 1, 0, 2)), NaN (1, 1, 1, 2))
%!assert (mean (ones (1, 1, 1, 0)), NaN (1, 1, 1, 1))
%!assert (mean (ones (1, 1, 2, 0)), NaN (1, 1, 1, 0))
%!assert (mean (ones (1, 2, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mean (ones (1, 2, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mean (ones (1, 2, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mean (ones (1, 2, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mean (ones (1, 2, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mean (ones (2, 0, 0, 0)), NaN (1, 0, 0, 0))
%!assert (mean (ones (2, 0, 0, 1)), NaN (1, 0, 0, 1))
%!assert (mean (ones (2, 0, 0, 2)), NaN (1, 0, 0, 2))
%!assert (mean (ones (2, 0, 1, 0)), NaN (1, 0, 1, 0))
%!assert (mean (ones (2, 0, 1, 1)), NaN (1, 0, 1, 1))
%!assert (mean (ones (2, 0, 1, 2)), NaN (1, 0, 1, 2))
%!assert (mean (ones (2, 0, 2, 0)), NaN (1, 0, 2, 0))
%!assert (mean (ones (2, 0, 2, 1)), NaN (1, 0, 2, 1))
%!assert (mean (ones (2, 0, 2, 2)), NaN (1, 0, 2, 2))
%!assert (mean (ones (2, 1, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mean (ones (2, 1, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mean (ones (2, 1, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mean (ones (2, 1, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mean (ones (2, 1, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mean (ones (2, 2, 0, 0)), NaN (1, 2, 0, 0))
%!assert (mean (ones (2, 2, 0, 1)), NaN (1, 2, 0, 1))
%!assert (mean (ones (2, 2, 0, 2)), NaN (1, 2, 0, 2))
%!assert (mean (ones (2, 2, 1, 0)), NaN (1, 2, 1, 0))
%!assert (mean (ones (2, 2, 2, 0)), NaN (1, 2, 2, 0))
%!assert (mean (ones (1, 1, 0, 0, 0)), NaN (1, 1, 1, 0, 0))
%!assert (mean (ones (1, 1, 1, 1, 0)), NaN (1, 1, 1, 1, 1))
%!assert (mean (ones (2, 1, 1, 1, 0)), NaN (1, 1, 1, 1, 0))
%!assert (mean (ones (1, 2, 1, 1, 0)), NaN (1, 1, 1, 1, 0))
%!assert (mean (ones (1, 3, 0, 2)), NaN (1, 1, 0, 2)) 
%!assert (mean (single (ones (1, 3, 0, 2))), single (NaN (1, 1, 0, 2)))
%!assert (mean ([], 1), NaN (1, 0))
%!assert (mean ([], 2), NaN (0, 1))
%!assert (mean ([], 3), [])
%!assert (mean (ones (1, 0), 1), NaN (1, 0))
%!assert (mean (ones (1, 0), 2), NaN)
%!assert (mean (ones (1, 0), 3), NaN (1, 0))
%!assert (mean (ones (0, 1), 1), NaN)
%!assert (mean (ones (0, 1), 2), NaN (0, 1))
%!assert (mean (ones (0, 1), 3), NaN (0, 1))

## Test input validation
%!error mean ()
%!error mean (1, 2, 3, 4)
%!error <X must be a numeric> mean ({1:5})
%!error <OPT must be a string> mean (1, 2, 3)
%!error <DIM must be an integer> mean (1, ones (2, 2))
%!error <DIM must be an integer> mean (1, 1.5)
%!error <DIM must be .* a valid dimension> mean (1, 0)
%!error <X must not contain any negative values> mean ([1 -1], "g")
%!error <option 'b' not recognized> mean (1, "b")
