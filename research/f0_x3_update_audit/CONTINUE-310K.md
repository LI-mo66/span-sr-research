# X3 frontcap 300K to 310K

Requires the completed 300K manager report and verified candidate checkpoint.
Run this external manager using the unchanged pinned 6e71b81 candidate tree.
Original runner, architecture, Adam moments, rho=0.0005, gradient clip=0.1,
LR=1e-4 and original 200/400/600/800K milestones remain unchanged.
Resume full state, evaluate ordinary and x8 every 1K, same quality stops
relative to healthy original 295K. Maximum 310K; no automatic shutdown.
This is staged evidence gathering, not a guarantee of long-run stability.
Manager may resume completed 1K boundaries in 300K..310K after review,
using a fresh stage directory; anomalies are never silently removed.
