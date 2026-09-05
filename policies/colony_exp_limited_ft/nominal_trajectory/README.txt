Nominal published Colony policy rollout (not a separately optimized path).

Load using drive_trajopt.traj.load_artifact(directory).
Plot using: python -m drive_trajopt.traj DIRECTORY
Files: traj.csv, surface_track.npz, meta.yaml, stats.json, plots/.
The track surface is the original Colony corridor. There is no pulled-back
surface_ref: use active="track" with s_track/e_track/dphi_track.
All units are SI; dphi is course-heading error, not body-heading error.
Commands on row k apply to the following interval; final row holds the last
command. measured_* columns retain the incoming measured controls.
Initial vx=0.1 m/s, s=0, e=0, beta=0, previous torque=250 Nm.
The startup transient is retained; this is one complete nominal rollout.
Only the final crossing is linearly interpolated to the exact lap-end s.
See meta.yaml for exact policy hashes, dynamics checkpoint and ENU origin.
This trajectory has not been validated on the real car.
