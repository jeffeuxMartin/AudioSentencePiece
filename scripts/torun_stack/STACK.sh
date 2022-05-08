
# ============== constant ============== #
WORKDIR=$HOME/AudioWords
APROOT=$WORKDIR/AudioSentencePiece
RUNNING=$APROOT/scripts/run_battleship.sh

# ---------------- main ---------------- #
cd $WORKDIR
cd AudioSentencePiece; git pull; cd ..
hrun -c 8 -m 16 -GG -g 3090 zsh $RUNNING -b $((18 / 2)) --original           --run_name 'BART ASR'
hrun -c 8 -m 16 -GG -g 3090 zsh $RUNNING -b $((18 / 2))                      --run_name 'CIF ASR'
hrun -c 8 -m 16 -GG -g 3090 zsh $RUNNING -b $((18 / 2)) --original --notcoll --run_name 'BART ASR uncollapsed units'
hrun -c 8 -m 16 -GG -g 3090 zsh $RUNNING -b $((18 / 2))            --notcoll --run_name 'CIF ASR uncollapsed units'

# parser.add_argument("--run_name", type=str, default=None)
# parser.add_argument("-b", "--batch_size", type=int, default=6)
# parser.add_argument("-lr", "--lr", type=float, default=2e-4)
# parser.add_argument("-e", "--epochs", type=int, default=10)
# parser.add_argument("--eval_steps", type=int, default=500)
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--vram", type=float, default=10)

# parser.add_argument("--weight_len", type=float, default=None)
# parser.add_argument("--notcoll", 
#     action='store_false', dest='coll')
# parser.set_defaults(coll=True)

# parser.add_argument("--nowandb", 
#     action='store_false', dest='wandb')
# parser.set_defaults(wandb=True)

# parser.add_argument("--nolower", 
#     action='store_false', dest='lower')
# parser.set_defaults(lower=True)

# parser.add_argument('--fix_encoder', action='store_true')
# parser.add_argument('--original', action='store_true')
# parser.add_argument('--autoencoder', action='store_true')

# args = parser.parse_args()

# batch_scaled_up = max(int(args.vram / 10.), 1)
# # args.batch_size *= batch_scaled_up
# # if batch_scaled_up > 1:
# #     logging.warning(
# #         f"Batch size resized to {args.batch_size:3d}..."
# #     )

# default_run_name = (
#     # f"lr = {args.lr}, bsz = {args.batch_size} ({batch_scaled_up} scaled_up)"
#     f"lr = {args.lr}, bsz = {args.batch_size}, {args.epochs} epochs"
#     + (" (coll)" if args.coll else " (orig)")
#     + (" (lower)" if args.lower else " (normalcase)")
#     + (" (fix_encoder)" if args.fix_encoder else "")
#     + (" (orignalTfm)" if args.original else "")
#     + (" (autoencoder)" if args.autoencoder else "")
#     + (f" weight_len = {args.weight_len}" if args.weight_len is not None else "")
# )
# if args.run_name is None:
#     args.run_name = default_run_name
# else:
#     args.run_name = args.run_name + " " + default_run_name
# return args
