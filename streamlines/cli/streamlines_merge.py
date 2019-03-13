from .. import io, Streamlines

def streamlines_merge(streamlines_to_merge, outfile):

    streamlines = [io.load(tract_file) for tract_file in streamlines_to_merge]

    merged = streamlines[0]

    for streamline in streamlines[1:]:
        merged += streamline

    io.save(merged, outfile)

