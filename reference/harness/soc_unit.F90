! Unit driver for individual SOCRATES radiance_core kernels.
! Usage: soc_unit <input.bin> <output.bin>
! input.bin (stream): char(32) mode, int32 n_profile, n_layer, iopt, then real64
!   arrays in the order listed for each mode below (Fortran order).
! output.bin: records as in dump_spectrum (char(40) name, kind, rank, dims, data).
program soc_unit
use realtype_rd, only: RealK
use rad_pcf
use def_control, only: StrCtrl
use def_bound, only: StrBound
use dump_io
use single_scattering_mod, only: single_scattering
use rescale_tau_omega_mod, only: rescale_tau_omega
use two_coeff_mod, only: two_coeff
use two_coeff_fast_lw_mod, only: two_coeff_fast_lw
use ir_source_mod, only: ir_source
use solar_source_mod, only: solar_source
use solver_homogen_direct_mod, only: solver_homogen_direct
use solver_no_scat_mod, only: solver_no_scat
use monochromatic_gas_flux_mod, only: monochromatic_gas_flux
implicit none

character(len=1024) :: fin, fout
character(len=32) :: mode
integer(4) :: np4, nl4, iopt4
integer :: np, nl, iopt, ui, ierr
type(StrCtrl) :: control
type(StrBound) :: bound
real(RealK), allocatable :: a(:,:), b(:,:), c(:,:), d(:,:), e(:,:)
real(RealK), allocatable :: v1(:), v2(:), v3(:)
real(RealK), allocatable :: o1(:,:), o2(:,:), o3(:,:), o4(:,:), sc(:,:,:), ft(:,:), fd(:,:)

call get_command_argument(1, fin)
call get_command_argument(2, fout)
open(newunit=ui, file=trim(fin), access='stream', form='unformatted', status='old')
read(ui) mode, np4, nl4, iopt4
np = np4; nl = nl4; iopt = iopt4
open(newunit=uo, file=trim(fout), access='stream', form='unformatted', status='replace')
allocate(a(np,nl), b(np,nl), c(np,nl), d(np,nl), e(np,nl), v1(np), v2(np), v3(np))
allocate(o1(np,nl), o2(np,nl), o3(np,nl), o4(np,nl), sc(np,nl,2), ft(np,2*nl+2), fd(np,0:nl))
ierr = i_normal

select case (trim(mode))

case ('single_scattering')          ! iopt = scatter method; k_grey_tot, k_ext_scat, k_gas_abs, d_mass
  read(ui) a, b, c, d
  call single_scattering(iopt, np, 1, nl, d, a, b, c, o1, o2, np, nl, 1, nl)
  call wr_r('tau', [np,nl], pack(o1,.true.)); call wr_r('omega', [np,nl], pack(o2,.true.))

case ('rescale_tau_omega')          ! tau, omega, forward_scatter
  read(ui) a, b, c
  call rescale_tau_omega(np, 1, nl, a, b, c, np, nl, 1)
  call wr_r('tau', [np,nl], pack(a,.true.)); call wr_r('omega', [np,nl], pack(b,.true.))

case ('two_coeff_solar')            ! iopt = i_2stream; asymmetry, omega, tau, sec_0(np)
  read(ui) a, b, c, v1
  control%isolir = ip_solar
  control%i_2stream = iopt
  control%l_spherical_solar = .false.
  call two_coeff(ierr, control, np, 1, nl, iopt, a, b, c, c, ip_solar, v1, d, &
    o1, o2, o3, o4, sc, np, 1, nl, 1, nl, 2)
  call wr_r('trans', [np,nl], pack(o1,.true.)); call wr_r('reflect', [np,nl], pack(o2,.true.))
  call wr_r('trans_0', [np,nl], pack(o4,.true.))
  call wr_r('source_up', [np,nl], pack(sc(:,:,ip_scf_solar_up),.true.))
  call wr_r('source_down', [np,nl], pack(sc(:,:,ip_scf_solar_down),.true.))

case ('two_coeff_fast_lw')          ! iopt = l_ir_source_quad (0/1); tau
  read(ui) a
  sc = 0.0_RealK
  call two_coeff_fast_lw(np, 1, nl, iopt /= 0, a, o1, sc, np, nl, 1, nl, 2)
  call wr_r('trans', [np,nl], pack(o1,.true.))
  call wr_r('source_1', [np,nl], pack(sc(:,:,ip_scf_ir_1d),.true.))
  call wr_r('source_2', [np,nl], pack(sc(:,:,ip_scf_ir_2d),.true.))

case ('ir_source')                  ! iopt = quad; source_1, source_2, diff_planck, diff_planck_2
  read(ui) a, b, c, d
  sc(:,:,1) = a; sc(:,:,2) = b
  call ir_source(np, 1, nl, sc, c, iopt /= 0, d, o1, o2, np, nl, 2)
  call wr_r('s_down', [np,nl], pack(o1,.true.)); call wr_r('s_up', [np,nl], pack(o2,.true.))

case ('solar_source')               ! iopt = l_scale_solar; flux_inc_direct(np), trans_0, source_up, source_down, adjust_solar_ke
  read(ui) v1, a, b, c, d
  control%l_orog = .false.
  sc(:,:,ip_scf_solar_up) = b; sc(:,:,ip_scf_solar_down) = c
  call solar_source(control, bound, np, nl, v1, a, a, sc, iopt /= 0, d, fd, o1, o2, np, nl, 2)
  call wr_r('flux_direct', [np,nl+1], pack(fd,.true.))
  call wr_r('s_down', [np,nl], pack(o1,.true.)); call wr_r('s_up', [np,nl], pack(o2,.true.))

case ('solver_homogen_direct')      ! trans, reflect, s_down, s_up, diffuse_albedo(np), flux_inc_down(np), source_ground(np)
  read(ui) a, b, c, d, v1, v2, v3
  call solver_homogen_direct(np, nl, a, b, c, d, v1, v2, v3, ft, np, nl)
  call wr_r('flux_total', [np,2*nl+2], pack(ft,.true.))

case ('solver_no_scat')             ! trans, s_down, s_up, diffuse_albedo(np), flux_inc_down(np), d_planck_flux_surface(np)
  read(ui) a, b, c, v1, v2, v3
  call solver_no_scat(np, nl, a, b, c, v1, v2, v3, ft, np, nl)
  call wr_r('flux_total', [np,2*nl+2], pack(ft,.true.))

case ('monochromatic_gas_flux_ir')  ! tau_gas, diff_planck, flux_inc_down(np), d_planck_flux_surface(np), diffuse_albedo(np)
  read(ui) a, b, v1, v2, v3
  call monochromatic_gas_flux(np, nl, a, ip_infra_red, v1, v1, v1, b, v2, v3, v3, &
    1.66_RealK, fd, ft, np, nl)
  call wr_r('flux_total', [np,2*nl+2], pack(ft,.true.))

case default
  write(0,*) 'unknown mode ', trim(mode)
  stop 2
end select
if (ierr /= i_normal) then
  write(0,*) 'ierr = ', ierr
  stop 3
end if
close(uo)
close(ui)
end program soc_unit
