! Tracing replacement for SOCRATES' dummy DR_HOOK: prints each routine entry
! to unit 0 so the exact call tree for a configuration can be recorded.
MODULE yomhook
  USE parkind1, ONLY: jpim, jprb
  IMPLICIT NONE
  LOGICAL, PARAMETER :: lhook = .TRUE.
CONTAINS
  SUBROUTINE dr_hook(name,code,handle)
    IMPLICIT NONE
    CHARACTER(len=*),   INTENT(in)    :: name
    INTEGER(kind=jpim), INTENT(in)    :: code
    REAL(kind=jprb),    INTENT(inout) :: handle
    IF (code == 0) WRITE(0,'(A)') 'ENTER '//TRIM(name)
  END SUBROUTINE dr_hook
END MODULE yomhook
