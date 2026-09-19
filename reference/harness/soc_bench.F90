! Timing driver: SOCRATES as called by Isca's socrates_interface (chunks of
! chunk_size columns through socrates_calc's sequence), timing only the
! radiation calculation (no file I/O, no spectral-file reading).
!
! Usage: soc_bench <spectral_file> <input.bin (soc_ref format)> <chunk_size>
! After setup it prints READY; then for each line "go" on stdin it runs the whole
! input once and prints "TIME <seconds> <checksum>"; "quit" ends.
program soc_bench
use realtype_rd,  only: RealK
use rad_pcf
use def_control,  only: StrCtrl
use def_spectrum, only: StrSpecData
use def_dimen,    only: StrDim
use def_atm,      only: StrAtm,   deallocate_atm
use def_bound,    only: StrBound, deallocate_bound
use def_cld,      only: StrCld,   deallocate_cld, deallocate_cld_prsc
use def_aer,      only: StrAer,   deallocate_aer, deallocate_aer_prsc
use def_out,      only: StrOut,   deallocate_out
use read_control_mod, only: read_control
use set_control_mod,  only: set_control
use set_dimen_mod,    only: set_dimen
use set_atm_mod,      only: set_atm
use set_bound_mod,    only: set_bound
use socrates_set_cld, only: set_cld
use set_aer_mod,      only: set_aer
use socrates_config_mod
use soc_constants_mod, only: i_def
implicit none

interface
  subroutine radiance_calc(control, dimen, spectrum, atm, cld, aer, bound, radout)
    use def_control,  only: StrCtrl
    use def_spectrum, only: StrSpecData
    use def_dimen,    only: StrDim
    use def_atm,      only: StrAtm
    use def_cld,      only: StrCld
    use def_aer,      only: StrAer
    use def_bound,    only: StrBound
    use def_out,      only: StrOut
    type(StrCtrl),     intent(in)  :: control
    type(StrDim),      intent(in)  :: dimen
    type(StrSpecData), intent(in)  :: spectrum
    type(StrAtm),      intent(in)  :: atm
    type(StrCld),      intent(in)  :: cld
    type(StrAer),      intent(in)  :: aer
    type(StrBound),    intent(in)  :: bound
    type(StrOut),      intent(out) :: radout
  end subroutine radiance_calc
end interface

character(len=1024) :: spectral_file, in_file, arg, cmd
integer :: u, ios, i0, i1   ! chunk_size is Isca's own socrates_rad_nml variable
integer(4) :: isolir, n_profile, n_layer, do_clouds_int
integer(8) :: c0, c1, rate
logical :: do_clouds
type(StrCtrl) :: control
type(StrSpecData) :: spectrum
real(RealK), allocatable :: p_layer(:,:), t_layer(:,:), t_level(:,:), d_mass(:,:), &
  density(:,:), h2o(:,:), o3(:,:), co2(:,:), heat_capacity(:,:), &
  cld_frac(:,:), reff(:,:), mmr_cl(:,:)
real(RealK), allocatable :: t_surf(:), coszen(:), solar_irrad(:), albedo(:), orog_corr(:)
real(RealK) :: emissivity, checksum

call get_command_argument(1, spectral_file)
call get_command_argument(2, in_file)
call get_command_argument(3, arg)
read(arg, *) chunk_size

open(newunit=u, file=trim(in_file), access='stream', form='unformatted', status='old', action='read')
read(u) isolir, n_profile, n_layer, do_clouds_int
do_clouds = (do_clouds_int /= 0)
allocate(p_layer(n_profile,n_layer), t_layer(n_profile,n_layer), t_level(n_profile,0:n_layer), &
  d_mass(n_profile,n_layer), density(n_profile,n_layer), h2o(n_profile,n_layer), &
  o3(n_profile,n_layer), co2(n_profile,n_layer), heat_capacity(n_profile,n_layer), &
  cld_frac(n_profile,n_layer), reff(n_profile,n_layer), mmr_cl(n_profile,n_layer))
allocate(t_surf(n_profile), coszen(n_profile), solar_irrad(n_profile), albedo(n_profile), orog_corr(n_profile))
read(u) p_layer, t_layer, t_level, d_mass, density, h2o, o3, co2, &
  t_surf, coszen, solar_irrad, albedo, emissivity, heat_capacity, cld_frac, reff, mmr_cl
close(u)
orog_corr = 0.0_RealK
call read_spectrum(trim(spectral_file), spectrum)

write(*, '(a)') 'READY'
flush(6)
call system_clock(count_rate=rate)
do
  read(*, '(a)', iostat=ios) cmd
  if (ios /= 0) exit
  if (trim(cmd) /= 'go') exit
  checksum = 0.0_RealK
  call system_clock(c0)
  ! --- socrates_interface: read_control once per call, then chunks ---
  control%isolir = isolir
  call read_control(control, spectrum, do_clouds)
  do i0 = 1, n_profile, chunk_size
    i1 = min(i0 + chunk_size - 1, n_profile)
    call calc_chunk(i0, i1)
  end do
  call system_clock(c1)
  write(*, '(a, es24.16, 1x, es24.16)') 'TIME ', real(c1 - c0, 8) / real(rate, 8), checksum
  flush(6)
end do

contains

subroutine calc_chunk(i0, i1)
  integer, intent(in) :: i0, i1
  type(StrDim) :: dimen
  type(StrAtm) :: atm
  type(StrBound) :: bound
  type(StrCld) :: cld
  type(StrAer) :: aer
  type(StrOut) :: radout
  integer(i_def) :: np8, nl8
  real(RealK) :: zeros(i1-i0+1, n_layer)
  np8 = i1 - i0 + 1
  nl8 = n_layer
  zeros = 0.0_RealK
  ! --- socrates_calc.F90 ---
  call set_control(control)
  call set_dimen(control, dimen, spectrum, np8, nl8, nl8, nl8, nl8, nl8)
  call set_atm(control, dimen, spectrum, atm, np8, nl8, p_layer(i0:i1,:), t_layer(i0:i1,:), &
    t_level(i0:i1,:), d_mass(i0:i1,:), density(i0:i1,:), h2o(i0:i1,:), o3(i0:i1,:), co2(i0:i1,:))
  call set_bound(control, dimen, spectrum, bound, np8, t_surf(i0:i1), coszen(i0:i1), &
    solar_irrad(i0:i1), orog_corr(i0:i1), l_planet_grey_surface, albedo(i0:i1), emissivity)
  call set_cld(cld, control, dimen, spectrum, np8, nl8, liq_frac=cld_frac(i0:i1,:), ice_frac=zeros, &
    liq_mmr=mmr_cl(i0:i1,:), ice_mmr=zeros, liq_dim=reff(i0:i1,:), ice_dim=zeros)
  call set_aer(control, dimen, spectrum, aer, np8)
  call radiance_calc(control, dimen, spectrum, atm, cld, aer, bound, radout)
  checksum = checksum + sum(radout%flux_up(1:np8, 0:n_layer, 1)) + sum(radout%flux_down(1:np8, 0:n_layer, 1))
  call deallocate_out(radout)
  call deallocate_aer_prsc(aer)
  call deallocate_aer(aer)
  call deallocate_cld_prsc(cld)
  call deallocate_cld(cld)
  call deallocate_bound(bound)
  call deallocate_atm(atm)
end subroutine calc_chunk
end program soc_bench
