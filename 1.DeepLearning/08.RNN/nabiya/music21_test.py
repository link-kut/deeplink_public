import music21

conv_midi = music21.converter.subConverters.ConverterMidi()

#conv_midi.parseData("tinynotation: 3/4 c4 d8 f g16 a g f#")

m = music21.converter.parse("tinynotation: 2/4 g8 e8 e8 e4 d8 d8 e8 d8 d8 e8 f8 d8 e8 e8 d8 e8 e8 f8 g8 g4 g8 e8 e8 e8 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4 d8 d8 d8 d8 d8 e8 f4 e8 e8 e8 e8 e8 f8 g4 g8 e8 e4 f8 d8 d4 ")

#fp = m.write('midi', fp='pathToWhereYouWantToWriteIt')

m.show("midi")