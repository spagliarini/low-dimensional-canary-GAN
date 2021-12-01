

class Annotation(object):

    @staticmethod
    def _slice_audio(audio, features, feature_rate=None):
        if feature_rate is None:
            feature_rate = len(audio) // len(features)

        slices = []
        for i in range(0, feature_rate * len(features), feature_rate):
            slices.append((i, i + feature_rate - 1))

        if len(audio) % feature_rate != 0:
            slices.append(((len(audio) // feature_rate) * feature_rate, -1))

        return slices

    def __init__(self, audio, input_id, features, vects, vocab, feature_rate):

        self.id = input_id
        self.audio = audio
        self.feat = features
        self.vect = vects
        self.vocab = vocab

        self._feature_rate = feature_rate
        self._audio_lookup = Annotation._slice_audio(audio, features, feature_rate)

    def __repr__(self):
        s = f"<Annotation> id: {self.id} seq: "
        for l in self.lbl:
            s += l + " "
        return s

    def __len__(self):
        return len(self.vect)

    def __getitem__(self, val):
        if type(val) is slice:
            audio_idx = slice(self._audio_lookup[val][0][0], self._audio_lookup[val][-1][-1])
            return Annotation(self.audio[audio_idx], self.id, self.feat[val],
                          self.vect[val], self.vocab, self._feature_rate)
        else:
            audio_idx = slice(*self._audio_lookup[val])
            return Annotation(self.audio[audio_idx], self.id, [self.feat[val]],
                              [self.vect[val]], self.vocab, self._feature_rate)

    def __iter__(self):
        for i in range(self.__len__()):
            audio_idx = slice(self._audio_lookup[i][0], self._audio_lookup[i][1])
            yield Annotation(self.audio[audio_idx], self.id, [self.feat[i]],
                          [self.vect[i]], self.vocab, self._feature_rate)

    @property
    def lbl(self):
        return [self.vocab[v.argmax()] for v in self.vect]
