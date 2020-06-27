import copy


class ScoreRunner:
    def __init__(self, score):
        self.score = score

    def advance(self, seconds):
        return self.score._advance(seconds)


class Score:
    def _advance(self, seconds):
        """Request the score element to advance some seconds.

        Never call this function directly. Let ScoreRunner do it for you.
        The idea is that Score elements are "immutable" so you can do crazy compositions.

        We mutate them and track state only when running copies of them.

        Returns:
            finished: Boolean
            advanced: Nunber of seconds consumed.
                      If not finished, advances must equal seconds
        """
        pass

    def __add__(self, other):
        return Sequence([self, other])

    def __mul__(self, times):
        return Repeat(self, times)

    def run(self):
        """Start running this score."""
        return ScoreRunner(self.copy())

    def copy(self):
        """Return a copy that we can run."""
        return copy.copy(self)


class Call(Score):
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def _advance(self, seconds):
        self.function(*self.args, **self.kwargs)
        return True, 0


class Wait(Score):
    def __init__(self, seconds):
        self.wait_period = seconds
        self.elapsed = 0

    def _advance(self, seconds):
        if self.elapsed + seconds >= self.wait_period:
            return True, self.wait_period - self.elapsed

        self.elapsed += seconds
        return False, seconds


class Sequence(Score):
    def __init__(self, scores):
        self.scores = scores
        self.score_ptr = 0

    def _advance(self, seconds):
        if self.score_ptr >= len(self.scores):
            return True, 0

        total_elapsed = 0
        while total_elapsed <= seconds:
            finished, elapsed = self.scores[self.score_ptr]._advance(seconds - total_elapsed)
            total_elapsed += elapsed

            if finished:
                self.score_ptr += 1

            # we are done
            if self.score_ptr >= len(self.scores):
                return True, total_elapsed

            if total_elapsed == seconds and not finished:
                break

        return False, total_elapsed

    def copy(self):
        new_scores = [s.copy() for s in self.scores]
        return Sequence(new_scores)


class Repeat(Score):
    def __init__(self, score, times):
        self.base_score = score
        self.score = None
        self.score_ptr = 0
        self.times = times

    def _advance(self, seconds):
        if self.score_ptr >= self.times:
            return True, 0

        total_elapsed = 0

        if self.score is None:
            self.score = self.base_score.copy()

        while total_elapsed <= seconds:
            finished, elapsed = self.score._advance(seconds - total_elapsed)
            total_elapsed += elapsed

            if finished:
                self.score = self.base_score.copy()
                self.score_ptr += 1

            # we are done
            if self.score_ptr >= self.times:
                return True, total_elapsed

            if total_elapsed == seconds and not finished:
                break

        return False, total_elapsed


class TimeWarp(Score):
    def __init__(self, score, warp_function, warp_inverse):
        self.score = score
        self.warp_function = warp_function
        self.warp_inverse = warp_inverse
        self.timer = 0

    def _advance(self, seconds):
        f_advance = self.warp_function(self.timer + seconds) - self.warp_function(self.timer)
        finished, elapsed = self.score._advance(f_advance)

        if not finished:
            self.timer += seconds
            return True, self.timer
        else:
            return False, self.warp_inverse(
                self.warp_function(self.timer) + elapsed
            ) - self.timer
