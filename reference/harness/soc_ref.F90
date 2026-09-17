! Reference driver: runs the SOCRATES core exactly as Isca's socrates_calc does
! (Isca read_control / set_control / set_dimen / set_atm / set_bound / set_cld /
! set_aer), but reads inputs from and writes outputs to raw binary stream files
! so results can be compared against the PyTorch port.
!
! Usage: soc_ref <spectral_file> <input.bin> <output.bin> [namelist_file]
!
! input.bin  (little-endian stream):
!   int32  isolir, n_profile, n_layer, do_clouds
!   real64 p_layer(np,nl), t_layer(np,nl), t_level(np,0:nl), d_mass(np,nl),
!          density(np,nl), h2o(np,nl), o3(np,nl), co2(np,nl),
!          t_surf(np), coszen(np), solar_irrad(np), albedo(np), emissivity,
!          heat_capacity(np,nl), cld_frac(np,nl), reff(np,nl), mmr_cl(np,nl)
! output.bin:
!   int32  n_profile, n_layer, n_band
!   real64 flux_direct, flux_down, flux_up (np,0:nl)
!          flux_direct_clear, flux_down_clear, flux_up_clear (np,0:nl)
!          heating_rate (np,nl)
!          flux_direct_clear_band, flux_down_clear_band, flux_up_clear_band (np,0:nl,nb)
!          tot_cloud_cover (np)
program soc_ref

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
use read_control_mod,    only: read_control
use set_control_mod,     only: set_control
use set_dimen_mod,       only: set_dimen
use set_atm_mod,         only: set_atm
use set_bound_mod,       only: set_bound
use socrates_set_cld,    only: set_cld
use set_aer_mod,         only: set_aer
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

character(len=1024) :: spectral_file, in_file, out_file, nml_file
integer :: nargs, u, ios
integer(4) :: isolir, n_profile, n_layer, do_clouds_int, n_band
logical :: do_clouds
integer :: i, l
integer(i_def) :: np8, nl8

type(StrCtrl) :: control
type(StrSpecData) :: spectrum
type(StrDim) :: dimen
type(StrAtm) :: atm
type(StrBound) :: bound
type(StrCld) :: cld
type(StrAer) :: aer
type(StrOut) :: radout

real(RealK), allocatable :: p_layer(:,:), t_layer(:,:), t_level(:,:), d_mass(:,:), &
  density(:,:), h2o(:,:), o3(:,:), co2(:,:), heat_capacity(:,:), &
  cld_frac(:,:), reff(:,:), mmr_cl(:,:), zeros_cld(:,:)
real(RealK), allocatable :: t_surf(:), coszen(:), solar_irrad(:), albedo(:), orog_corr(:)
real(RealK) :: emissivity
real(RealK), allocatable :: heating_rate(:,:), tot_cloud_cover(:)

nargs = command_argument_count()
if (nargs < 3) then
  write(0,*) 'usage: soc_ref <spectral_file> <input.bin> <output.bin> [namelist]'
  stop 2
end if
call get_command_argument(1, spectral_file)
call get_command_argument(2, in_file)
call get_command_argument(3, out_file)
if (nargs >= 4) then
  call get_command_argument(4, nml_file)
  open(newunit=u, file=trim(nml_file), status='old', action='read')
  read(u, nml=socrates_rad_nml, iostat=ios)
  if (ios /= 0) then
    write(0,*) 'error reading socrates_rad_nml, iostat=', ios
    stop 3
  end if
  close(u)
end if

open(newunit=u, file=trim(in_file), access='stream', form='unformatted', status='old', action='read')
read(u) isolir, n_profile, n_layer, do_clouds_int
do_clouds = (do_clouds_int /= 0)
allocate(p_layer(n_profile,n_layer), t_layer(n_profile,n_layer), t_level(n_profile,0:n_layer), &
  d_mass(n_profile,n_layer), density(n_profile,n_layer), h2o(n_profile,n_layer), &
  o3(n_profile,n_layer), co2(n_profile,n_layer), heat_capacity(n_profile,n_layer), &
  cld_frac(n_profile,n_layer), reff(n_profile,n_layer), mmr_cl(n_profile,n_layer), &
  zeros_cld(n_profile,n_layer))
allocate(t_surf(n_profile), coszen(n_profile), solar_irrad(n_profile), albedo(n_profile), &
  orog_corr(n_profile))
read(u) p_layer, t_layer, t_level, d_mass, density, h2o, o3, co2, &
  t_surf, coszen, solar_irrad, albedo, emissivity, heat_capacity, cld_frac, reff, mmr_cl
close(u)
orog_corr = 0.0_RealK
zeros_cld = 0.0_RealK

call read_spectrum(trim(spectral_file), spectrum)
n_band = spectrum%basic%n_band

control%isolir = isolir
call read_control(control, spectrum, do_clouds)

! --- as Isca socrates_calc ---
call set_control(control)
control%l_flux_direct_clear_band = .true.   ! extra diagnostic only

np8 = n_profile
nl8 = n_layer
call set_dimen(control, dimen, spectrum, np8, nl8, &
  nl8, nl8, nl8, nl8)
call set_atm(control, dimen, spectrum, atm, np8, nl8, &
  p_layer, t_layer, t_level, d_mass, density, h2o, o3, co2)
call set_bound(control, dimen, spectrum, bound, np8, &
  t_surf, coszen, solar_irrad, orog_corr, &
  l_planet_grey_surface, albedo, emissivity)
call set_cld(cld, control, dimen, spectrum, np8, nl8, &
  liq_frac = cld_frac, ice_frac = zeros_cld, liq_mmr = mmr_cl, &
  ice_mmr = zeros_cld, liq_dim = reff, ice_dim = zeros_cld)
call set_aer(control, dimen, spectrum, aer, np8)

call radiance_calc(control, dimen, spectrum, atm, cld, aer, bound, radout)

allocate(heating_rate(n_profile, n_layer), tot_cloud_cover(n_profile))
do l = 1, n_profile
  do i = 1, n_layer
    heating_rate(l, i) = (radout%flux_down(l,i-1,1) - radout%flux_down(l,i,1) &
      + radout%flux_up(l,i,1) - radout%flux_up(l,i-1,1)) / heat_capacity(l, i)
  end do
end do
tot_cloud_cover = 0.0_RealK
if (control%l_cloud) tot_cloud_cover = radout%tot_cloud_cover(1:n_profile)

open(newunit=u, file=trim(out_file), access='stream', form='unformatted', status='replace', action='write')
write(u) n_profile, n_layer, n_band
write(u) radout%flux_direct(1:n_profile,0:n_layer,1), radout%flux_down(1:n_profile,0:n_layer,1), &
  radout%flux_up(1:n_profile,0:n_layer,1)
write(u) radout%flux_direct_clear(1:n_profile,0:n_layer,1), radout%flux_down_clear(1:n_profile,0:n_layer,1), &
  radout%flux_up_clear(1:n_profile,0:n_layer,1)
write(u) heating_rate
write(u) radout%flux_direct_clear_band(1:n_profile,0:n_layer,1:n_band), &
  radout%flux_down_clear_band(1:n_profile,0:n_layer,1:n_band), &
  radout%flux_up_clear_band(1:n_profile,0:n_layer,1:n_band)
write(u) tot_cloud_cover
close(u)

end program soc_ref
