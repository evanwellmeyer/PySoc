! Dump the Wenyi CO2/O3 k-scaling tables (scale_wenyi.F90) for the Python port.
program dump_wenyi
use scale_wenyi, only: plg, ttb, tto, gk250b, gk4, gk6
use dump_io
implicit none
character(len=1024) :: oname
call get_command_argument(1, oname)
open(newunit=uo, file=trim(oname), access='stream', form='unformatted', status='replace')
call wr_r('plg', shape(plg), plg)
call wr_r('ttb', shape(ttb), ttb)
call wr_r('tto', shape(tto), tto)
call wr_r('gk250b', shape(gk250b), gk250b)
call wr_r('gk4', shape(gk4), pack(gk4, .true.))
call wr_r('gk6', shape(gk6), pack(gk6, .true.))
close(uo)
end program dump_wenyi
