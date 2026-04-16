remote_path="u1192515@notchpeak1.chpc.utah.edu:/uufs/chpc.utah.edu/common/home/u1192515/dev/aeon"
remote_path_="u1192515@notchpeak1.chpc.utah.edu:/uufs/chpc.utah.edu/common/home/u1192515/dev/aeon_"


pull_output_flags=(
    "--include=output/"
    "--include=output/**"
    "--exclude=**"
)

pull_output_flags_=(
    "--include=output_/"
    "--include=output_/**"
    "--exclude=**"
)

push_src_flags=(
    "--exclude=output/"
    "--exclude=output/**"
    "--exclude=output_/"
    "--exclude=output_/**"
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

push_history_flags_=(
    "--include=output_/"
    "--include=output_/axicombo/"
    "--include=output_/axicombo/*/"
    "--include=output_/axicombo/*/search/"
    "--include=output_/axicombo/*/search/history.csv"
    "--exclude=**"
)

rsync_pull_output() {
    rsync -avz --relative $pull_output_flags $* $remote_path/./ ./
}

rsync_pull_output_() {
    rsync -avz --relative $pull_output_flags_ $* $remote_path_/./ ./
}

rsync_push_src() {
    rsync -avz --relative $push_src_flags $* ./ $remote_path
}

rsync_push_src_() {
    rsync -avz --relative $push_src_flags $* ./ $remote_path_
}

rsync_push_history() {
    rsync -avz --relative $push_history_flags $* ./ $remote_path
}

rsync_push_history_() {
    rsync -avz --relative $push_history_flags_ $* ./ $remote_path_
}
