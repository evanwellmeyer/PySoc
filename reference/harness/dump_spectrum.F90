! Dump the arrays filled by SOCRATES read_spectrum, to validate the Python
! spectral-file reader. Record format (stream, little endian):
!   char(40) name, int32 kind (1=int, 2=real64), int32 rank, int32 dims(rank), data
! Usage: dump_spectrum <spectral_file> <out.bin>

program dump_spectrum
use realtype_rd, only: RealK
use def_spectrum, only: StrSpecData
use dump_io
implicit none
type(StrSpecData) :: sp
character(len=1024) :: fname, oname

call get_command_argument(1, fname)
call get_command_argument(2, oname)
call read_spectrum(trim(fname), sp)
open(newunit=uo, file=trim(oname), access='stream', form='unformatted', status='replace')

call wr_i('n_band', [1], [sp%basic%n_band])
call wr_i('n_absorb', [1], [sp%gas%n_absorb])
call wr_i('l_present', [size(sp%basic%l_present)], merge(1, 0, sp%basic%l_present))
call wr_r('wavelength_short', shape(sp%basic%wavelength_short), sp%basic%wavelength_short)
call wr_r('wavelength_long', shape(sp%basic%wavelength_long), sp%basic%wavelength_long)
call wr_i('n_band_exclude', shape(sp%basic%n_band_exclude), sp%basic%n_band_exclude)
if (allocated(sp%basic%index_exclude)) &
  call wr_i('index_exclude', shape(sp%basic%index_exclude), pack(sp%basic%index_exclude, .true.))
if (sp%basic%l_present(2)) &
  call wr_r('solar_flux_band', shape(sp%solar%solar_flux_band), sp%solar%solar_flux_band)
if (sp%basic%l_present(3)) then
  call wr_i('i_rayleigh_scheme', [1], [sp%rayleigh%i_rayleigh_scheme])
  call wr_r('rayleigh_coeff', shape(sp%rayleigh%rayleigh_coeff), sp%rayleigh%rayleigh_coeff)
end if
call wr_i('type_absorb', shape(sp%gas%type_absorb), sp%gas%type_absorb)
call wr_i('n_band_absorb', shape(sp%gas%n_band_absorb), sp%gas%n_band_absorb)
call wr_i('index_absorb', shape(sp%gas%index_absorb), pack(sp%gas%index_absorb, .true.))
call wr_i('i_overlap', shape(sp%gas%i_overlap), sp%gas%i_overlap)
call wr_i('i_band_k', shape(sp%gas%i_band_k), pack(sp%gas%i_band_k, .true.))
call wr_i('i_scale_k', shape(sp%gas%i_scale_k), pack(sp%gas%i_scale_k, .true.))
call wr_i('i_scale_fnc', shape(sp%gas%i_scale_fnc), pack(sp%gas%i_scale_fnc, .true.))
call wr_i('i_scat', shape(sp%gas%i_scat), pack(sp%gas%i_scat, .true.))
call wr_r('k', shape(sp%gas%k), pack(sp%gas%k, .true.))
call wr_r('w', shape(sp%gas%w), pack(sp%gas%w, .true.))
call wr_r('scale', shape(sp%gas%scale), pack(sp%gas%scale, .true.))
call wr_r('p_ref', shape(sp%gas%p_ref), pack(sp%gas%p_ref, .true.))
call wr_r('t_ref', shape(sp%gas%t_ref), pack(sp%gas%t_ref, .true.))
call wr_i('num_ref_p', shape(sp%gas%num_ref_p), pack(sp%gas%num_ref_p, .true.))
call wr_i('num_ref_t', shape(sp%gas%num_ref_t), pack(sp%gas%num_ref_t, .true.))
call wr_i('index_sb', shape(sp%gas%index_sb), sp%gas%index_sb)
if (allocated(sp%gas%p_lookup)) then
  call wr_r('p_lookup', shape(sp%gas%p_lookup), sp%gas%p_lookup)
  call wr_r('t_lookup', shape(sp%gas%t_lookup), pack(sp%gas%t_lookup, .true.))
  call wr_r('k_lookup', shape(sp%gas%k_lookup), pack(sp%gas%k_lookup, .true.))
end if
if (sp%basic%l_present(6)) then
  call wr_i('n_deg_fit', [1], [sp%planck%n_deg_fit])
  call wr_r('t_ref_planck', [1], [sp%planck%t_ref_planck])
  call wr_i('l_planck_tbl', [1], [merge(1, 0, sp%planck%l_planck_tbl)])
  call wr_r('thermal_coeff', shape(sp%planck%thermal_coeff), pack(sp%planck%thermal_coeff, .true.))
end if
if (sp%basic%l_present(8)) then
  call wr_i('n_band_continuum', shape(sp%cont%n_band_continuum), sp%cont%n_band_continuum)
  call wr_i('index_continuum', shape(sp%cont%index_continuum), pack(sp%cont%index_continuum, .true.))
  call wr_i('index_water', [1], [sp%cont%index_water])
end if
if (sp%basic%l_present(9)) then
  call wr_i('i_scale_fnc_cont', shape(sp%cont%i_scale_fnc_cont), pack(sp%cont%i_scale_fnc_cont, .true.))
  call wr_r('k_cont', shape(sp%cont%k_cont), pack(sp%cont%k_cont, .true.))
  call wr_r('scale_cont', shape(sp%cont%scale_cont), pack(sp%cont%scale_cont, .true.))
  call wr_r('p_ref_cont', shape(sp%cont%p_ref_cont), pack(sp%cont%p_ref_cont, .true.))
  call wr_r('t_ref_cont', shape(sp%cont%t_ref_cont), pack(sp%cont%t_ref_cont, .true.))
end if
if (sp%basic%l_present(10)) then
  call wr_i('drop_l_type', shape(sp%drop%l_drop_type), merge(1, 0, sp%drop%l_drop_type))
  call wr_i('drop_i_parm', shape(sp%drop%i_drop_parm), sp%drop%i_drop_parm)
  call wr_i('drop_n_phf', shape(sp%drop%n_phf), sp%drop%n_phf)
  call wr_r('drop_parm_list', shape(sp%drop%parm_list), pack(sp%drop%parm_list, .true.))
  call wr_r('drop_parm_min_dim', shape(sp%drop%parm_min_dim), sp%drop%parm_min_dim)
  call wr_r('drop_parm_max_dim', shape(sp%drop%parm_max_dim), sp%drop%parm_max_dim)
end if
if (sp%basic%l_present(12)) then
  call wr_i('ice_l_type', shape(sp%ice%l_ice_type), merge(1, 0, sp%ice%l_ice_type))
  call wr_i('ice_i_parm', shape(sp%ice%i_ice_parm), sp%ice%i_ice_parm)
  call wr_i('ice_n_phf', shape(sp%ice%n_phf), sp%ice%n_phf)
  call wr_r('ice_parm_list', shape(sp%ice%parm_list), pack(sp%ice%parm_list, .true.))
  call wr_r('ice_parm_min_dim', shape(sp%ice%parm_min_dim), sp%ice%parm_min_dim)
  call wr_r('ice_parm_max_dim', shape(sp%ice%parm_max_dim), sp%ice%parm_max_dim)
end if
close(uo)
end program dump_spectrum
