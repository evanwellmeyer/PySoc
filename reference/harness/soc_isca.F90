! Reference for the Isca-level interface: the preprocessing of Isca's
! socrates_interface.F90 / run_socrates (copied, clear sky, no time stepping or
! astronomy) around two socrates_calc-equivalent calls (LW, then SW with the
! temperature updated by the LW heating over delta_t).
!
! Usage: soc_isca <sp_lw> <sp_sw> <input.bin> <output.bin> [namelist]
! input.bin: int32 n_profile, n_layer; real64 temp(np,nl), q(np,nl), p_full(np,nl),
!   p_half(np,nl+1), z_full(np,nl), z_half(np,nl+1), t_surf(np), albedo(np), coszen(np),
!   ozone(np,nl) [mmr], rrsun, delta_t, int32 do_clouds,
!   [if do_clouds: cf_rad(np,nl), reff_rad(np,nl) [m], qcl_rad(np,nl) [kg/kg]]
! output.bin: records (dump_io format)
module isca_consts
  implicit none
  ! Isca constants_mod
  real(8), parameter :: grav = 9.80d0, rdgas = 287.04d0, kappa = 2.d0/7.d0
  real(8), parameter :: cp_air = rdgas/kappa, wtmco2 = 44.00995d0, gas_constant = 8.314d0
end module isca_consts

program soc_isca
use realtype_rd, only: RealK
use rad_pcf
use def_control,  only: StrCtrl
use def_spectrum, only: StrSpecData
use def_dimen,    only: StrDim
use def_atm,      only: StrAtm
use def_bound,    only: StrBound
use def_cld,      only: StrCld
use def_aer,      only: StrAer
use def_out,      only: StrOut
use read_control_mod, only: read_control
use set_control_mod,  only: set_control
use set_dimen_mod,    only: set_dimen
use set_atm_mod,      only: set_atm
use set_bound_mod,    only: set_bound
use socrates_set_cld, only: set_cld
use set_aer_mod,      only: set_aer
use socrates_config_mod
use soc_constants_mod, only: i_def
use isca_consts
use dump_io
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

character(len=1024) :: sp_lw_file, sp_sw_file, fin, fout, fnml
integer :: ui, ios, nargs
integer(4) :: np4, nl4
integer :: n_profile, n_layer, i
real(8), allocatable :: temp(:,:), q(:,:), p_full(:,:), p_half(:,:), z_full(:,:), z_half(:,:), &
  t_surf(:), albedo(:), coszen(:), ozone(:,:), co2_in(:,:), tg_tmp(:,:)
real(8) :: rrsun, delta_t
integer(4) :: do_clouds4
logical :: do_clouds
real(8), allocatable :: cf_rad(:,:), reff_rad(:,:), qcl_rad(:,:), mmr_cl(:,:)
real(8), allocatable :: up_lw_clr(:,:), dn_lw_clr(:,:), up_sw_clr(:,:), dn_sw_clr(:,:), cover_lw(:), cover_sw(:)
real(8), allocatable :: hr_lw(:,:), hr_sw(:,:), up_lw(:,:), dn_lw(:,:), up_sw(:,:), dn_sw(:,:), &
  t_half_lw(:,:), t_half_sw(:,:), spectral_olr(:,:)
type(StrSpecData) :: sp_lw, sp_sw

nargs = command_argument_count()
call get_command_argument(1, sp_lw_file)
call get_command_argument(2, sp_sw_file)
call get_command_argument(3, fin)
call get_command_argument(4, fout)
if (nargs >= 5) then
  call get_command_argument(5, fnml)
  open(newunit=ui, file=trim(fnml), status='old')
  read(ui, nml=socrates_rad_nml, iostat=ios)
  close(ui)
  if (ios /= 0) stop 3
end if

open(newunit=ui, file=trim(fin), access='stream', form='unformatted', status='old')
read(ui) np4, nl4
n_profile = np4; n_layer = nl4
allocate(temp(n_profile,n_layer), q(n_profile,n_layer), p_full(n_profile,n_layer), &
  p_half(n_profile,n_layer+1), z_full(n_profile,n_layer), z_half(n_profile,n_layer+1), &
  t_surf(n_profile), albedo(n_profile), coszen(n_profile), ozone(n_profile,n_layer), &
  co2_in(n_profile,n_layer), tg_tmp(n_profile,n_layer))
read(ui) temp, q, p_full, p_half, z_full, z_half, t_surf, albedo, coszen, ozone, rrsun, delta_t
allocate(cf_rad(n_profile,n_layer), reff_rad(n_profile,n_layer), qcl_rad(n_profile,n_layer), mmr_cl(n_profile,n_layer))
read(ui) do_clouds4
do_clouds = (do_clouds4 /= 0)
if (do_clouds) then
  read(ui) cf_rad, reff_rad, qcl_rad
  mmr_cl = qcl_rad / (1.0 - qcl_rad)   ! run_socrates
else
  cf_rad = 0.; reff_rad = 0.; mmr_cl = 0.
end if
close(ui)

call read_spectrum(trim(sp_lw_file), sp_lw)
call read_spectrum(trim(sp_sw_file), sp_sw)
allocate(hr_lw(n_profile,n_layer), hr_sw(n_profile,n_layer), up_lw(n_profile,0:n_layer), &
  dn_lw(n_profile,0:n_layer), up_sw(n_profile,0:n_layer), dn_sw(n_profile,0:n_layer), &
  t_half_lw(n_profile,0:n_layer), t_half_sw(n_profile,0:n_layer), spectral_olr(n_profile, sp_lw%basic%n_band))
allocate(up_lw_clr(n_profile,0:n_layer), dn_lw_clr(n_profile,0:n_layer), up_sw_clr(n_profile,0:n_layer), &
  dn_sw_clr(n_profile,0:n_layer), cover_lw(n_profile), cover_sw(n_profile))

! run_socrates: CO2 from co2_ppmv
if (input_co2_mmr .eqv. .false.) then
  co2_in = co2_ppmv * 1.e-6 * wtmco2 / (1000. * gas_constant / rdgas )
else
  co2_in = co2_ppmv * 1.e-6
end if

tg_tmp = temp
call interface_call(.true., sp_lw, tg_tmp, hr_lw, up_lw, dn_lw, t_half_lw, up_lw_clr, dn_lw_clr, cover_lw)
tg_tmp = tg_tmp + hr_lw*delta_t
call interface_call(.false., sp_sw, tg_tmp, hr_sw, up_sw, dn_sw, t_half_sw, up_sw_clr, dn_sw_clr, cover_sw)

open(newunit=uo, file=trim(fout), access='stream', form='unformatted', status='replace')
call wr_r('tdt_lw', [n_profile, n_layer], pack(hr_lw, .true.))
call wr_r('tdt_sw', [n_profile, n_layer], pack(hr_sw, .true.))
call wr_r('flux_lw_up', [n_profile, n_layer+1], pack(up_lw, .true.))
call wr_r('flux_lw_down', [n_profile, n_layer+1], pack(dn_lw, .true.))
call wr_r('flux_sw_up', [n_profile, n_layer+1], pack(up_sw, .true.))
call wr_r('flux_sw_down', [n_profile, n_layer+1], pack(dn_sw, .true.))
call wr_r('t_half_lw', [n_profile, n_layer+1], pack(t_half_lw, .true.))
call wr_r('t_half_sw', [n_profile, n_layer+1], pack(t_half_sw, .true.))
call wr_r('spectral_olr', [n_profile, sp_lw%basic%n_band], pack(spectral_olr, .true.))
call wr_r('co2', [n_profile, n_layer], pack(co2_in, .true.))
call wr_r('flux_lw_up_clear', [n_profile, n_layer+1], pack(up_lw_clr, .true.))
call wr_r('flux_lw_down_clear', [n_profile, n_layer+1], pack(dn_lw_clr, .true.))
call wr_r('flux_sw_up_clear', [n_profile, n_layer+1], pack(up_sw_clr, .true.))
call wr_r('flux_sw_down_clear', [n_profile, n_layer+1], pack(dn_sw_clr, .true.))
call wr_r('tot_cloud_cover_lw', [n_profile], cover_lw)
call wr_r('tot_cloud_cover_sw', [n_profile], cover_sw)
close(uo)

contains

subroutine interface_call(soc_lw_mode, spectrum, fms_temp, heating, flux_up, flux_down, t_level, &
  flux_up_clr, flux_down_clr, cover)
  logical, intent(in) :: soc_lw_mode
  type(StrSpecData), intent(in) :: spectrum
  real(8), intent(in) :: fms_temp(:,:)
  real(8), intent(out) :: heating(:,:), flux_up(:,0:), flux_down(:,0:), t_level(:,0:)
  real(8), intent(out) :: flux_up_clr(:,0:), flux_down_clr(:,0:), cover(:)
  type(StrCtrl) :: control
  type(StrDim) :: dimen
  type(StrAtm) :: atm
  type(StrBound) :: bound
  type(StrCld) :: cld
  type(StrAer) :: aer
  type(StrOut) :: radout
  real(8), dimension(n_profile, n_layer) :: input_t, input_p, input_mixing_ratio, input_o3, &
    input_d_mass, input_density, input_heat_capacity, zeros
  real(8), dimension(n_profile, 0:n_layer) :: input_p_level, z_half_reshaped
  real(8), dimension(n_profile) :: input_t_surf, input_coszen, input_solar_irrad, input_orog, input_albedo
  integer(i_def) :: np8, nl8
  integer :: i

  ! --- socrates_interface.F90 ---
  input_t = fms_temp
  input_p = p_full
  input_p_level = p_half
  if (account_for_effect_of_water .eqv. .true.) then
    input_mixing_ratio = q / (1. - q)
  else
    input_mixing_ratio = 0.0
  end if
  if (account_for_effect_of_ozone .eqv. .true.) then
    input_o3 = ozone
  else
    input_o3 = 0.0
  end if
  input_coszen = coszen
  input_orog = 0.0
  input_albedo = albedo
  input_solar_irrad = stellar_constant * rrsun
  input_t_surf = t_surf
  z_half_reshaped = z_half
  call interp_temp(z_full, z_half_reshaped, input_t, t_level)
  do i = n_layer, 1, -1
    input_d_mass(:,i) = (input_p_level(:,i)-input_p_level(:,i-1))/grav
    input_density(:,i) = input_p(:,i)/(rdgas*input_t(:,i))
    input_heat_capacity(:,i) = input_d_mass(:,i)*cp_air
  end do
  if (soc_lw_mode) then
    control%isolir = ip_infra_red
  else
    control%isolir = ip_solar
  end if
  call read_control(control, spectrum, do_clouds)

  ! --- socrates_calc.F90 ---
  np8 = n_profile; nl8 = n_layer
  zeros = 0.0
  call set_control(control)
  call set_dimen(control, dimen, spectrum, np8, nl8, nl8, nl8, nl8, nl8)
  call set_atm(control, dimen, spectrum, atm, np8, nl8, input_p, input_t, t_level, &
    input_d_mass, input_density, input_mixing_ratio, input_o3, co2_in)
  call set_bound(control, dimen, spectrum, bound, np8, input_t_surf, input_coszen, &
    input_solar_irrad, input_orog, l_planet_grey_surface, input_albedo, input_planet_emissivity)
  call set_cld(cld, control, dimen, spectrum, np8, nl8, liq_frac=cf_rad, ice_frac=zeros, &
    liq_mmr=mmr_cl, ice_mmr=zeros, liq_dim=reff_rad, ice_dim=zeros)
  call set_aer(control, dimen, spectrum, aer, np8)
  call radiance_calc(control, dimen, spectrum, atm, cld, aer, bound, radout)
  do i = 1, n_layer
    heating(:, i) = (radout%flux_down(1:n_profile,i-1,1)-radout%flux_down(1:n_profile,i,1) &
      + radout%flux_up(1:n_profile,i,1)-radout%flux_up(1:n_profile,i-1,1)) / input_heat_capacity(:, i)
  end do
  flux_up_clr = radout%flux_up_clear(1:n_profile, 0:n_layer, 1)
  flux_down_clr = radout%flux_down_clear(1:n_profile, 0:n_layer, 1)
  cover = 0.0
  if (control%l_cloud) cover = radout%tot_cloud_cover(1:n_profile)
  flux_up = radout%flux_up(1:n_profile, 0:n_layer, 1)
  flux_down = radout%flux_down(1:n_profile, 0:n_layer, 1)
  if (soc_lw_mode) spectral_olr = radout%flux_up_clear_band(1:n_profile, 0, 1:spectrum%basic%n_band)
end subroutine interface_call

! Verbatim from Isca socrates_interface.F90 (indices of t_half are 1-based there).
subroutine interp_temp(z_full,z_half,temp_in, t_half)
    implicit none

    real(8),dimension(:,:),intent(in)  :: z_full,z_half,temp_in
    real(8),dimension(size(z_half,1), size(z_half,2)),intent(out) :: t_half

    integer i,k,kend
    real(8) dzk,dzk1,dzk2

    kend=size(z_full,2)
    do k=2,kend
        do i=1,size(temp_in,1)
            dzk2 = 1./( z_full(i,k-1)   - z_full(i,k) )
            dzk  = ( z_half(i,k  )   - z_full(i,k) )*dzk2
            dzk1 = ( z_full(i,k-1)   - z_half(i,k) )*dzk2
            t_half(i,k) = temp_in(i,k)*dzk1 + temp_in(i,k-1)*dzk
        enddo
    enddo
    do i=1,size(temp_in,1)
        t_half(i,1) = 0.5*(3*temp_in(i,1)-temp_in(i,2))
        t_half(i,kend+1) = temp_in(i,kend-1) &
            + (z_half(i,kend+1) - z_full(i,kend-1))  &
            * (temp_in(i,kend ) - temp_in(i,kend-1)) &
            / (z_full(i,kend  ) - z_full(i,kend-1))
    enddo

end subroutine interp_temp
end program soc_isca
