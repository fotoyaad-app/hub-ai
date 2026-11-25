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
# Mean / std of (cos(pos) - cos(neg)) over 180 random songs of the library (25 Nov 2026, larger_clap_music_and_speech).
# Stored score = sigmoid(2 * (z - 0.9)): 0.5 = top ~18%, 0.55 = top ~16% (the Hub's boost starts there), 0.9 = clearly that mood.
MOOD_CAL = {"calm": [-0.0116, 0.1455], "party": [0.0722, 0.1442], "devotional": [-0.181, 0.2116], "sad": [-0.0864, 0.1115], "romantic": [0.1001, 0.1803], "focus": [-0.3105, 0.1876], "sleep": [-0.0558, 0.1348]}
MUSIC_TAG_GROUPS = {
 'genre': ['carnatic classical music', 'hindustani classical music', 'indian folk music', 'indian film song', 'devotional music', 'hip hop', 'rap', 'electronic dance music', 'rock', 'pop', 'jazz', 'lo-fi', 'ghazal', 'qawwali', 'western classical music'],
}
