! Minimal stand-in for the FMS error handling used by Isca's socrates_set_cld.
module fms_mod
implicit none
integer, parameter :: FATAL = 2, WARNING = 1, NOTE = 0
contains
subroutine error_mesg(routine, message, level)
  character(len=*), intent(in) :: routine, message
  integer, intent(in) :: level
  write(0, '(a)') trim(routine)//': '//trim(message)
  if (level == FATAL) stop 1
end subroutine error_mesg
end module fms_mod
