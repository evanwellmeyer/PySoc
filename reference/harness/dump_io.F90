! Record writer shared by the reference harness programs.
module dump_io
use realtype_rd, only: RealK
implicit none
integer :: uo
contains
subroutine wr_i(name, dims, v)
  character(len=*), intent(in) :: name
  integer, intent(in) :: dims(:), v(:)
  character(len=40) :: nm
  nm = name
  write(uo) nm, int(1,4), int(size(dims),4), int(dims,4), int(v,4)
end subroutine
subroutine wr_r(name, dims, v)
  character(len=*), intent(in) :: name
  integer, intent(in) :: dims(:)
  real(RealK), intent(in) :: v(:)
  character(len=40) :: nm
  nm = name
  write(uo) nm, int(2,4), int(size(dims),4), int(dims,4), real(v,8)
end subroutine
end module dump_io
