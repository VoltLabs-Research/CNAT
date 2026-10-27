# Bulk metal snapshots for the multi-structure CNAT model (FCC, HCP, BCC, liquid). Same protocol as
# in.bulk: 5 ps NVT at tinit (0.3 Tm for crystals, 2.5 Tm to melt liquids), 5 ps NVT at temp, 10 ps NPT
# at temp and 0 bar, then two further 5 ps NPT legs for the held-out and extra snapshots. Crystals use
# anisotropic NPT so HCP relaxes its c/a; liquids use isotropic NPT.
#
# lmp -in in.metal -var name mg.hcp.400K -var lat hcp -var a0 3.1842 -var nx 28 -var ny 16 -var nz 17 \
#     -var style eam/fs -var pfile Mg_mm.eam.fs -var elem Mg -var tinit 275 -var temp 400 -var baro aniso -var seed 7

log             log.${name}

units           metal
boundary        p p p
atom_style      atomic

lattice         ${lat} ${a0}
region          box block 0 ${nx} 0 ${ny} 0 ${nz}
create_box      1 box
create_atoms    1 box

pair_style      ${style}
pair_coeff      * * ${pfile} ${elem}

thermo          1000
thermo_style    custom step time temp pe press vol lx ly lz
thermo_modify   flush yes

variable        dt equal 0.001
timestep        ${dt}
variable        tdamp equal 100*${dt}
variable        pdamp equal 1000*${dt}

velocity        all create ${tinit} ${seed} mom yes rot yes dist gaussian

fix             nvt1 all nvt temp ${tinit} ${tinit} ${tdamp}
run             5000
unfix           nvt1

fix             nvt2 all nvt temp ${temp} ${temp} ${tdamp}
run             5000
unfix           nvt2

fix             npt1 all npt temp ${temp} ${temp} ${tdamp} ${baro} 0.0 0.0 ${pdamp}
run             10000
write_data      data.${name} nocoeff

run             5000
write_data      data.${name}.heldout nocoeff

run             5000
write_data      data.${name}.extra nocoeff
