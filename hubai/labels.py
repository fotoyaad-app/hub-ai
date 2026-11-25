# Zero-shot photo tags (SigLIP). Indian family-photo vocabulary; search uses these words.
PHOTO_TAGS = [
 'baby', 'child', 'kids playing', 'family photo', 'group photo', 'couple', 'selfie', 'portrait', 'old person', 'school', 'classroom',
 'wedding', 'bride', 'groom', 'haldi ceremony', 'mehendi', 'engagement ceremony', 'reception stage', 'baby shower', 'naming ceremony', 'cradle ceremony',
 'birthday party', 'birthday cake', 'cake cutting', 'balloons', 'anniversary', 'house warming', 'puja', 'temple', 'idol', 'diya lamp', 'rangoli', 'flower garland',
 'diwali', 'fireworks', 'holi colours', 'sankranti', 'kite', 'ganesh idol', 'bathukamma', 'christmas tree', 'eid', 'church', 'mosque',
 'saree', 'lehenga', 'kurta', 'dhoti', 'school uniform', 'gold jewellery', 'sunglasses',
 'food', 'biryani', 'dosa', 'idli', 'thali meal', 'sweets', 'ice cream', 'tea', 'coffee', 'fruits', 'restaurant', 'cooking', 'kitchen',
 'beach', 'sea', 'river', 'waterfall', 'lake', 'mountain', 'hill station', 'forest', 'garden', 'flowers', 'tree', 'farm', 'paddy field', 'village', 'sunset', 'sunrise', 'sky', 'clouds', 'rain', 'snow',
 'city street', 'building', 'apartment', 'house', 'living room', 'bedroom', 'office', 'hotel room', 'shopping mall', 'market', 'airport', 'aeroplane', 'train', 'railway station', 'bus', 'car', 'bike', 'scooter', 'auto rickshaw', 'road trip', 'bridge',
 'monument', 'fort', 'palace', 'statue', 'museum', 'zoo', 'park', 'playground', 'swimming pool', 'amusement park', 'stadium', 'cricket', 'football', 'gym', 'yoga',
 'dog', 'cat', 'cow', 'bird', 'elephant', 'monkey', 'horse', 'fish',
 'document', 'receipt', 'bill', 'id card', 'certificate', 'handwritten note', 'whiteboard', 'screenshot', 'chat screenshot', 'meme', 'poster', 'book', 'laptop', 'mobile phone', 'tv', 'computer screen',
 'night', 'indoor', 'outdoor', 'crowd', 'stage performance', 'dance', 'music concert', 'graduation', 'hospital', 'medicine', 'gift', 'shopping bags', 'money', 'car interior', 'mirror selfie',
]
PROMPT = 'a photo of {}.'

# CLAP music moods: each mood = (positive prompt, contrast prompt); score = P(positive) in a 2-way softmax.
MUSIC_MOODS = {
 'calm': ('calm, soft and relaxing music', 'loud, energetic and intense music'),
 'party': ('energetic dance party music with a heavy beat', 'slow, calm and quiet music'),
 'devotional': ('indian devotional prayer song, bhajan or temple chant', 'indian movie song about love or dance'),
 'sad': ('sad, emotional and melancholic song', 'happy, cheerful and upbeat song'),
 'romantic': ('romantic love song, a soft duet', 'aggressive, loud dance or rap track'),
 'focus': ('instrumental background music without singing, for studying', 'a song with lead vocals and lyrics'),
 'sleep': ('very slow, gentle lullaby for falling asleep', 'fast, upbeat and loud music'),
}
MUSIC_TAG_GROUPS = {
 'genre': ['carnatic classical music', 'hindustani classical music', 'indian folk music', 'indian film song', 'devotional music', 'hip hop', 'rap', 'electronic dance music', 'rock', 'pop', 'jazz', 'lo-fi', 'ghazal', 'qawwali', 'western classical music'],
 'vocals': ['male vocals', 'female vocals', 'male and female duet', 'chorus singing', 'no vocals, instrumental'],
 'instrument': ['flute', 'veena', 'sitar', 'piano', 'acoustic guitar', 'electric guitar', 'violin', 'tabla', 'drums', 'synthesizer', 'nadaswaram', 'mridangam'],
 'tempo': ['slow tempo', 'medium tempo', 'fast tempo'],
}
