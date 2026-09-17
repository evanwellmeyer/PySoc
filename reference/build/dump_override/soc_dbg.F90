! Debug dump of SOCRATES intermediates (enabled when env SOC_DUMP is set to a file path).
MODULE soc_dbg
USE realtype_rd, ONLY: RealK
IMPLICIT NONE
LOGICAL :: dbg_init = .FALSE., l_dbg = .FALSE.
INTEGER :: dbg_unit = -1
CONTAINS
SUBROUTINE dbg_start()
  CHARACTER(LEN=1024) :: path
  INTEGER :: n
  IF (dbg_init) RETURN
  dbg_init = .TRUE.
  CALL GET_ENVIRONMENT_VARIABLE('SOC_DUMP', path, n)
  IF (n > 0) THEN
    OPEN(NEWUNIT=dbg_unit, FILE=TRIM(path), ACCESS='stream', FORM='unformatted', STATUS='replace')
    l_dbg = .TRUE.
  END IF
END SUBROUTINE dbg_start
FUNCTION dbg_name(base, ib, ik) RESULT(nm)
  CHARACTER(LEN=*), INTENT(IN) :: base
  INTEGER, INTENT(IN) :: ib, ik
  CHARACTER(LEN=40) :: nm
  IF (ik >= 0) THEN
    WRITE(nm, '(a,"_b",i0,"_k",i0)') base, ib, ik
  ELSE
    WRITE(nm, '(a,"_b",i0)') base, ib
  END IF
END FUNCTION dbg_name
SUBROUTINE dbg_r(name, dims, v)
  CHARACTER(LEN=*), INTENT(IN) :: name
  INTEGER, INTENT(IN) :: dims(:)
  REAL(RealK), INTENT(IN) :: v(:)
  CHARACTER(LEN=40) :: nm
  CALL dbg_start()
  IF (.NOT. l_dbg) RETURN
  nm = name
  WRITE(dbg_unit) nm, INT(2,4), INT(SIZE(dims),4), INT(dims,4), REAL(v,8)
  FLUSH(dbg_unit)
END SUBROUTINE dbg_r
SUBROUTINE dbg_i(name, dims, v)
  CHARACTER(LEN=*), INTENT(IN) :: name
  INTEGER, INTENT(IN) :: dims(:), v(:)
  CHARACTER(LEN=40) :: nm
  CALL dbg_start()
  IF (.NOT. l_dbg) RETURN
  nm = name
  WRITE(dbg_unit) nm, INT(1,4), INT(SIZE(dims),4), INT(dims,4), INT(v,4)
  FLUSH(dbg_unit)
END SUBROUTINE dbg_i
END MODULE soc_dbg
