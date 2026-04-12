remote_path="u1192515@notchpeak1.chpc.utah.edu:/uufs/chpc.utah.edu/common/home/u1192515/dev/aeon"

pull_output_flags=(
    "--include=output/"
    "--include=output/**"
    "--exclude=**"
)

push_src_flags=(
    "--exclude=output/"
    "--exclude=output/**"
    "--exclude=Analyze/"
    "--exclude=Analyze/**"
    "--exclude=scripts/"
    "--exclude=scripts/**"
    "--exclude=target/"
    "--exclude=target/**"
    "--exclude=.git/"
    "--exclude=.git/**"
)

push_history_flags=(
    "--include=output/"
    "--include=output/axicombo/"
    "--include=output/axicombo/*/"
    "--include=output/axicombo/*/search/"
    "--include=output/axicombo/*/search/history.csv"
    "--exclude=**"
)

rsync_pull_output() {
    rsync -avz --relative $pull_output_flags $* $remote_path/./ ./
}

rsync_push_src() {
    rsync -avz --relative $push_src_flags $* ./ $remote_path
}

rsync_push_history() {
    rsync -avz --relative $push_history_flags $* ./ $remote_path
}
