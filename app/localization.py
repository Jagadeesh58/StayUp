"""Server-owned multilingual explanation templates.

The LLM never authors display text or facts. All numeric placeholders come from
validated state/evidence values; translations only alter surrounding wording.
"""
from __future__ import annotations
from typing import Any

LANGUAGES = ("English", "Hinglish", "Hindi", "Telugu", "Tamil", "Kannada", "Bengali")

CURRENT = {
"English": {
 "rate":"This chase needs {r} runs from {b} legal balls. Current rate: {cr}; required rate: {rr}. Required pace is {gap} runs per over {direction} the current rate. This is the equation, not a prediction.",
 "wickets":"There are {w} wickets remaining and {r} runs needed from {b} legal balls. Wickets in hand are a buffer, not a guarantee.",
 "runway":"The chase needs {r} runs from {b} legal balls, with {w} wickets remaining. The entered state cannot tell us what happened earlier or what happens next.",
 "comparison":"{a} leaves {ar} runs needed; {b} leaves {br}. That is {d} fewer runs under the second scenario. These are hypothetical six-legal-ball totals, not forecasts.",
 "rates":"After {a}, required rate is {ar}; after {b}, it is {br}. The second scenario's required rate is {direction} by {gap} runs per over. These are hypotheticals, not forecasts.",
 "wicket_comparison":"The {a} assumption leaves {aw} wickets; the {b} assumption leaves {bw}. The comparison is limited to the stated aggregate overs.",
 "target":"The final over began with {r} runs needed from {b} legal balls. The {deliveries} recorded deliveries added {added} runs, taking the score to {score} against target {target}. The source sequence reaches the target on its final recorded delivery.",
 "wicket_shift":"Across this verified sequence, wickets lost moved from {bw} to {aw}; required rate moved from {br} to {ar}. These are computed state changes, not player intent or a guarantee.",
 "rate_shift":"Required rate changed from {br} to {ar} across the selected source deliveries. This is a computed comparison, not a claim about what happens next.",
 "before_after":"Before this verified moment, {bruns} runs were needed from {bballs} balls. After it, {aruns} runs were needed from {aballs} balls. The source shows the equation change, not player intent."
},
"Hinglish": {
 "rate":"Chase ke liye {b} legal balls mein {r} runs chahiye. Current rate {cr} hai, required rate {rr} hai. Required pace current rate se {gap} runs/over {direction} hai. Yeh sirf equation hai, prediction nahi.",
 "wickets":"{b} legal balls mein {r} runs chahiye aur {w} wickets bache hain. Wickets buffer hain, guarantee nahi.",
 "runway":"{b} legal balls mein {r} runs chahiye, {w} wickets bache hain. Manual score se pehle kya hua ya aage kya hoga pata nahi chalta.",
 "comparison":"{a} ke baad {ar} runs chahiye; {b} ke baad {br}. Second scenario mein {d} runs kam chahiye. Dono six-legal-ball hypotheticals hain, forecast nahi.",
 "rates":"{a} ke baad required rate {ar}; {b} ke baad {br}. Second scenario ka rate {direction} {gap} runs/over hai. Yeh hypothetical hai, prediction nahi.",
 "wicket_comparison":"{a} assumption mein {aw} wickets bachte hain; {b} mein {bw}. Yeh sirf diye gaye over totals ka comparison hai.",
 "target":"Final over {b} legal balls mein {r} runs ki zaroorat se shuru hua. Recorded {deliveries} deliveries ne {added} runs banaye aur score {score} hua; target {target} tha. Source sequence mein target aakhri recorded delivery par reach hota hai.",
 "wicket_shift":"Verified sequence mein wickets {bw} se {aw} hui aur required rate {br} se {ar} hua. Yeh computed change hai, player intent ya guarantee nahi.",
 "rate_shift":"Selected deliveries mein required rate {br} se {ar} hua. Yeh source se nikla comparison hai, future ka claim nahi.",
 "before_after":"Is moment se pehle {bballs} balls mein {bruns} runs chahiye the. Baad mein {aballs} balls mein {aruns} runs chahiye the. Source equation ka change dikhata hai, player intent nahi."
},
"Hindi": {
 "rate":"इस चेज़ में {b} वैध गेंदों पर {r} रन चाहिए। वर्तमान रन रेट {cr} और आवश्यक रन रेट {rr} है। आवश्यक गति वर्तमान गति से {gap} रन प्रति ओवर {direction} है। यह समीकरण है, भविष्यवाणी नहीं।",
 "wickets":"{b} वैध गेंदों पर {r} रन चाहिए और {w} विकेट बाकी हैं। बचे हुए विकेट अवसर हैं, गारंटी नहीं।",
 "runway":"{b} वैध गेंदों पर {r} रन चाहिए और {w} विकेट बाकी हैं। दर्ज स्कोर से पहले की घटनाएँ या अगला परिणाम नहीं जाना जा सकता।",
 "comparison":"{a} के बाद {ar} रन चाहिए; {b} के बाद {br}। दूसरे परिदृश्य में {d} रन कम चाहिए। दोनों छह वैध गेंदों की काल्पनिक स्थितियाँ हैं, भविष्यवाणी नहीं।",
 "rates":"{a} के बाद आवश्यक रन रेट {ar}; {b} के बाद {br} है। दूसरे परिदृश्य का रन रेट {gap} रन प्रति ओवर {direction} है। यह काल्पनिक तुलना है।",
 "wicket_comparison":"{a} परिदृश्य में {aw} विकेट बचते हैं; {b} में {bw}। यह दिए गए ओवर सारांशों की तुलना है।",
 "target":"अंतिम ओवर की शुरुआत में {b} वैध गेंदों पर {r} रन चाहिए थे। {deliveries} दर्ज गेंदों पर {added} रन बने और स्कोर {score} हुआ; लक्ष्य {target} था। स्रोत क्रम में लक्ष्य आखिरी दर्ज गेंद पर पूरा हुआ।",
 "wicket_shift":"सत्यापित क्रम में खोए विकेट {bw} से {aw} हुए और आवश्यक रन रेट {br} से {ar} हुआ। यह गणना है, खिलाड़ी की मंशा या गारंटी नहीं।",
 "rate_shift":"चुनी गई स्रोत गेंदों के दौरान आवश्यक रन रेट {br} से {ar} हुआ। यह गणना है, आगे क्या होगा इसका दावा नहीं।",
 "before_after":"इस क्षण से पहले {bballs} गेंदों पर {bruns} रन चाहिए थे। बाद में {aballs} गेंदों पर {aruns} रन चाहिए थे। स्रोत समीकरण का बदलाव दिखाता है, मंशा नहीं।"
},
"Telugu": {
 "rate":"ఈ చేజ్‌లో {b} లీగల్ బంతుల్లో {r} పరుగులు కావాలి. ప్రస్తుత రన్‌రేట్ {cr}; అవసరమైన రన్‌రేట్ {rr}. అవసరమైన వేగం ప్రస్తుత వేగం కంటే ఓవర్‌కు {gap} పరుగులు {direction}. ఇది గణన మాత్రమే, అంచనా కాదు.",
 "wickets":"{b} లీగల్ బంతుల్లో {r} పరుగులు కావాలి; {w} వికెట్లు మిగిలాయి. మిగిలిన వికెట్లు అవకాశం మాత్రమే, హామీ కాదు.",
 "runway":"{b} లీగల్ బంతుల్లో {r} పరుగులు కావాలి; {w} వికెట్లు మిగిలాయి. నమోదు చేసిన స్కోరు గతంలో ఏమైందో లేదా తర్వాత ఏమవుతుందో చెప్పదు.",
 "comparison":"{a} తర్వాత {ar} పరుగులు కావాలి; {b} తర్వాత {br}. రెండో పరిస్థితిలో {d} పరుగులు తక్కువ కావాలి. ఇవి ఆరు లీగల్ బంతుల కల్పిత పరిస్థితులు, అంచనాలు కావు.",
 "rates":"{a} తర్వాత అవసరమైన రన్‌రేట్ {ar}; {b} తర్వాత {br}. రెండో పరిస్థితిలో రన్‌రేట్ ఓవర్‌కు {gap} పరుగులు {direction}. ఇది కల్పిత పోలిక మాత్రమే.",
 "wicket_comparison":"{a} పరిస్థితిలో {aw} వికెట్లు మిగులుతాయి; {b}లో {bw}. ఇది ఇచ్చిన ఓవర్ మొత్తాల పోలిక మాత్రమే.",
 "target":"చివరి ఓవర్ ప్రారంభంలో {b} లీగల్ బంతుల్లో {r} పరుగులు కావాలి. నమోదైన {deliveries} బంతుల్లో {added} పరుగులు వచ్చి స్కోరు {score} అయింది; లక్ష్యం {target}. మూల క్రమంలో చివరి నమోదైన బంతికి లక్ష్యం చేరింది.",
 "wicket_shift":"ధృవీకరించిన క్రమంలో కోల్పోయిన వికెట్లు {bw} నుంచి {aw}కు, అవసరమైన రన్‌రేట్ {br} నుంచి {ar}కు మారాయి. ఇది గణన మాత్రమే; ఆటగాడి ఉద్దేశం లేదా హామీ కాదు.",
 "rate_shift":"ఎంచుకున్న మూల బంతులలో అవసరమైన రన్‌రేట్ {br} నుంచి {ar}కి మారింది. ఇది గణించిన పోలిక, భవిష్యత్తు అంచనా కాదు.",
 "before_after":"ఈ క్షణానికి ముందు {bballs} బంతుల్లో {bruns} పరుగులు కావాలి. తర్వాత {aballs} బంతుల్లో {aruns} పరుగులు కావాలి. మూల సమాచారం స్కోరు సమీకరణ మార్పును చూపుతుంది, ఉద్దేశాన్ని కాదు."
},
"Tamil": {
 "rate":"இந்த சேஸில் {b} சட்டபூர்வ பந்துகளில் {r} ரன்கள் தேவை. தற்போதைய ரன் ரேட் {cr}; தேவையான ரன் ரேட் {rr}. தேவையான வேகம் தற்போதைய வேகத்தை விட ஓவருக்கு {gap} ரன்கள் {direction}. இது கணக்கு; கணிப்பு அல்ல.",
 "wickets":"{b} சட்டபூர்வ பந்துகளில் {r} ரன்கள் தேவை; {w} விக்கெட்டுகள் மீதம். மீதமுள்ள விக்கெட்டுகள் வாய்ப்பு மட்டுமே, உத்தரவாதம் அல்ல.",
 "runway":"{b} சட்டபூர்வ பந்துகளில் {r} ரன்கள் தேவை; {w} விக்கெட்டுகள் மீதம். உள்ளிடப்பட்ட ஸ்கோர் முன்பு என்ன நடந்தது அல்லது அடுத்து என்ன நடக்கும் என்பதைச் சொல்லாது.",
 "comparison":"{a} பிறகு {ar} ரன்கள் தேவை; {b} பிறகு {br}. இரண்டாவது சூழலில் {d} ரன்கள் குறைவாக தேவை. இரண்டும் ஆறு சட்டபூர்வ பந்துகளைக் கொண்ட கற்பனைச் சூழல்கள்; கணிப்பு அல்ல.",
 "rates":"{a} பிறகு தேவையான ரன் ரேட் {ar}; {b} பிறகு {br}. இரண்டாவது சூழலின் ரன் ரேட் ஓவருக்கு {gap} ரன்கள் {direction}. இது கற்பனை ஒப்பீடு.",
 "wicket_comparison":"{a} சூழலில் {aw} விக்கெட்டுகள் மீதம்; {b} சூழலில் {bw}. இது கொடுக்கப்பட்ட ஓவர் சுருக்கங்களின் ஒப்பீடு மட்டுமே.",
 "target":"கடைசி ஓவர் தொடங்கியபோது {b} சட்டபூர்வ பந்துகளில் {r} ரன்கள் தேவை. பதிவான {deliveries} பந்துகளில் {added} ரன்கள் எடுத்து ஸ்கோர் {score} ஆனது; இலக்கு {target}. மூல வரிசையில் கடைசி பதிவான பந்தில் இலக்கு எட்டப்பட்டது.",
 "wicket_shift":"சரிபார்க்கப்பட்ட வரிசையில் இழந்த விக்கெட்டுகள் {bw} இலிருந்து {aw} ஆகவும், தேவையான ரன் ரேட் {br} இலிருந்து {ar} ஆகவும் மாறின. இது கணக்கீடு; நோக்கம் அல்லது உத்தரவாதம் அல்ல.",
 "rate_shift":"தேர்ந்தெடுத்த மூல பந்துகளுக்கிடையில் தேவையான ரன் ரேட் {br} இலிருந்து {ar} ஆக மாறியது. இது கணக்கீடு; எதிர்காலக் கணிப்பு அல்ல.",
 "before_after":"இந்த தருணத்திற்கு முன் {bballs} பந்துகளில் {bruns} ரன்கள் தேவை. பின்னர் {aballs} பந்துகளில் {aruns} ரன்கள் தேவை. மூலத் தரவு சமன்பாட்டின் மாற்றத்தைக் காட்டுகிறது; நோக்கத்தைக் காட்டாது."
},
"Kannada": {
 "rate":"ಈ ಚೇಸ್‌ನಲ್ಲಿ {b} ಕಾನೂನುಬದ್ಧ ಎಸೆತಗಳಲ್ಲಿ {r} ರನ್ ಬೇಕು. ಪ್ರಸ್ತುತ ರನ್ ರೇಟ್ {cr}; ಅಗತ್ಯ ರನ್ ರೇಟ್ {rr}. ಅಗತ್ಯ ವೇಗ ಪ್ರಸ್ತುತ ವೇಗಕ್ಕಿಂತ ಓವರ್‌ಗೆ {gap} ರನ್ {direction}. ಇದು ಲೆಕ್ಕಾಚಾರ, ಭವಿಷ್ಯವಾಣಿ ಅಲ್ಲ.",
 "wickets":"{b} ಕಾನೂನುಬದ್ಧ ಎಸೆತಗಳಲ್ಲಿ {r} ರನ್ ಬೇಕು; {w} ವಿಕೆಟ್ ಉಳಿದಿವೆ. ಉಳಿದ ವಿಕೆಟ್‌ಗಳು ಅವಕಾಶ, ಖಾತರಿ ಅಲ್ಲ.",
 "runway":"{b} ಕಾನೂನುಬದ್ಧ ಎಸೆತಗಳಲ್ಲಿ {r} ರನ್ ಬೇಕು; {w} ವಿಕೆಟ್ ಉಳಿದಿವೆ. ನಮೂದಿಸಿದ ಸ್ಕೋರ್ ಹಿಂದಿನ ಘಟನೆಗಳನ್ನಾಗಲಿ ಮುಂದಿನ ಫಲಿತಾಂಶವನ್ನಾಗಲಿ ತಿಳಿಸುವುದಿಲ್ಲ.",
 "comparison":"{a} ನಂತರ {ar} ರನ್ ಬೇಕು; {b} ನಂತರ {br}. ಎರಡನೇ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ {d} ರನ್ ಕಡಿಮೆ ಬೇಕು. ಇವು ಆರು ಕಾನೂನುಬದ್ಧ ಎಸೆತಗಳ ಕಲ್ಪಿತ ಪರಿಸ್ಥಿತಿಗಳು; ಭವಿಷ್ಯವಾಣಿ ಅಲ್ಲ.",
 "rates":"{a} ನಂತರ ಅಗತ್ಯ ರನ್ ರೇಟ್ {ar}; {b} ನಂತರ {br}. ಎರಡನೇ ಪರಿಸ್ಥಿತಿಯ ರೇಟ್ ಓವರ್‌ಗೆ {gap} ರನ್ {direction}. ಇದು ಕಲ್ಪಿತ ಹೋಲಿಕೆ ಮಾತ್ರ.",
 "wicket_comparison":"{a} ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ {aw} ವಿಕೆಟ್ ಉಳಿಯುತ್ತವೆ; {b}ಯಲ್ಲಿ {bw}. ಇದು ನೀಡಿದ ಓವರ್ ಸಾರಾಂಶಗಳ ಹೋಲಿಕೆ.",
 "target":"ಕೊನೆಯ ಓವರ್ ಆರಂಭದಲ್ಲಿ {b} ಕಾನೂನುಬದ್ಧ ಎಸೆತಗಳಲ್ಲಿ {r} ರನ್ ಬೇಕಿತ್ತು. ದಾಖಲಾದ {deliveries} ಎಸೆತಗಳಲ್ಲಿ {added} ರನ್ ಬಂದು ಸ್ಕೋರ್ {score} ಆಯಿತು; ಗುರಿ {target}. ಮೂಲ ಕ್ರಮದಲ್ಲಿ ಗುರಿ ಕೊನೆಯ ದಾಖಲಾದ ಎಸೆತದಲ್ಲೇ ತಲುಪಿತು.",
 "wicket_shift":"ಪರಿಶೀಲಿಸಿದ ಕ್ರಮದಲ್ಲಿ ಕಳೆದುಕೊಂಡ ವಿಕೆಟ್‌ಗಳು {bw}ರಿಂದ {aw}ಕ್ಕೆ, ಅಗತ್ಯ ರನ್ ರೇಟ್ {br}ರಿಂದ {ar}ಕ್ಕೆ ಬದಲಾಯಿತು. ಇದು ಲೆಕ್ಕಾಚಾರ; ಆಟಗಾರನ ಉದ್ದೇಶ ಅಥವಾ ಖಾತರಿ ಅಲ್ಲ.",
 "rate_shift":"ಆಯ್ದ ಮೂಲ ಎಸೆತಗಳ ನಡುವೆ ಅಗತ್ಯ ರನ್ ರೇಟ್ {br}ರಿಂದ {ar}ಕ್ಕೆ ಬದಲಾಯಿತು. ಇದು ಲೆಕ್ಕಾಚಾರ, ಮುಂದಿನ ಫಲಿತಾಂಶದ ಹೇಳಿಕೆ ಅಲ್ಲ.",
 "before_after":"ಈ ಕ್ಷಣದ ಮೊದಲು {bballs} ಎಸೆತಗಳಲ್ಲಿ {bruns} ರನ್ ಬೇಕಿತ್ತು. ನಂತರ {aballs} ಎಸೆತಗಳಲ್ಲಿ {aruns} ರನ್ ಬೇಕಿತ್ತು. ಮೂಲ ಮಾಹಿತಿ ಸಮೀಕರಣದ ಬದಲಾವಣೆಯನ್ನು ತೋರಿಸುತ್ತದೆ, ಉದ್ದೇಶವನ್ನು ಅಲ್ಲ."
},
"Bengali": {
 "rate":"এই চেজে {b}টি বৈধ বলে {r} রান দরকার। বর্তমান রান রেট {cr}; প্রয়োজনীয় রান রেট {rr}। প্রয়োজনীয় গতি বর্তমান গতির চেয়ে প্রতি ওভারে {gap} রান {direction}। এটি হিসাব, ভবিষ্যদ্বাণী নয়।",
 "wickets":"{b}টি বৈধ বলে {r} রান দরকার; {w}টি উইকেট বাকি। বাকি উইকেট সুযোগ, নিশ্চয়তা নয়।",
 "runway":"{b}টি বৈধ বলে {r} রান দরকার; {w}টি উইকেট বাকি। দেওয়া স্কোর থেকে আগে কী হয়েছে বা পরে কী হবে তা জানা যায় না।",
 "comparison":"{a} এর পরে {ar} রান দরকার; {b} এর পরে {br}। দ্বিতীয় পরিস্থিতিতে {d} রান কম দরকার। দুটিই ছয়টি বৈধ বলের কাল্পনিক হিসাব, পূর্বাভাস নয়।",
 "rates":"{a} এর পরে প্রয়োজনীয় রান রেট {ar}; {b} এর পরে {br}। দ্বিতীয় পরিস্থিতির রেট প্রতি ওভারে {gap} রান {direction}। এটি কাল্পনিক তুলনা।",
 "wicket_comparison":"{a} পরিস্থিতিতে {aw}টি উইকেট বাকি; {b} তে {bw}। এটি দেওয়া ওভার-সারাংশের তুলনা।",
 "target":"শেষ ওভার শুরু হয়েছিল {b}টি বৈধ বলে {r} রান দরকার অবস্থায়। নথিভুক্ত {deliveries} বলে {added} রান হয়, স্কোর {score}; লক্ষ্য {target}। উৎসের ক্রমে শেষ নথিভুক্ত বলেই লক্ষ্য পূরণ হয়।",
 "wicket_shift":"যাচাইকৃত ধারায় হারানো উইকেট {bw} থেকে {aw} হয়েছে এবং প্রয়োজনীয় রান রেট {br} থেকে {ar} হয়েছে। এটি হিসাব, খেলোয়াড়ের উদ্দেশ্য বা নিশ্চয়তা নয়।",
 "rate_shift":"নির্বাচিত উৎস-বলগুলির মধ্যে প্রয়োজনীয় রান রেট {br} থেকে {ar} হয়েছে। এটি হিসাব, ভবিষ্যৎ ফলাফলের দাবি নয়।",
 "before_after":"এই মুহূর্তের আগে {bballs} বলে {bruns} রান দরকার ছিল। পরে {aballs} বলে {aruns} রান দরকার ছিল। উৎসের তথ্য সমীকরণের পরিবর্তন দেখায়, উদ্দেশ্য নয়।"
}
}

DIRECTION_WORDS={
 "English":{"above":"above","below":"below","higher":"higher","lower":"lower","the same":"the same"},
 "Hinglish":{"above":"zyada","below":"kam","higher":"zyada","lower":"kam","the same":"same"},
 "Hindi":{"above":"अधिक","below":"कम","higher":"अधिक","lower":"कम","the same":"समान"},
 "Telugu":{"above":"ఎక్కువ","below":"తక్కువ","higher":"ఎక్కువ","lower":"తక్కువ","the same":"అదే"},
 "Tamil":{"above":"அதிகம்","below":"குறைவு","higher":"அதிகம்","lower":"குறைவு","the same":"அதே"},
 "Kannada":{"above":"ಹೆಚ್ಚು","below":"ಕಡಿಮೆ","higher":"ಹೆಚ್ಚು","lower":"ಕಡಿಮೆ","the same":"ಅದೇ"},
 "Bengali":{"above":"বেশি","below":"কম","higher":"বেশি","lower":"কম","the same":"একই"},
}

def _rate(value: Any, language: str) -> str:
    if value is None:
        return {"English":"not defined","Hinglish":"define nahi hai","Hindi":"परिभाषित नहीं","Telugu":"నిర్వచించబడలేదు","Tamil":"வரையறுக்கப்படவில்லை","Kannada":"ವ್ಯಾಖ್ಯಾನಿಸಲಾಗಿಲ್ಲ","Bengali":"নির্ধারিত নয়"}.get(language,"not defined")
    return f"{float(value):.2f}"

def localized_explanation(intent: str, focus_id: str, context: dict[str, Any], language: str = "English") -> dict[str, Any]:
    language = language if language in LANGUAGES else "English"
    t = CURRENT[language]
    if intent == "current_state_context":
        s=context["state"]; r=int(s["runs_required"]); b=int(s["balls_remaining"]); w=int(s["wickets_remaining"])
        if focus_id == "rate_pressure" and s.get("current_run_rate") is not None and s.get("required_run_rate") is not None:
            gap=abs(float(s["required_run_rate"])-float(s["current_run_rate"]))
            direction=DIRECTION_WORDS[language]["above" if s["required_run_rate"]>=s["current_run_rate"] else "below"]
            answer=t["rate"].format(r=r,b=b,w=w,cr=_rate(s["current_run_rate"],language),rr=_rate(s["required_run_rate"],language),gap=f"{gap:.2f}",direction=direction)
        elif focus_id == "wickets_buffer": answer=t["wickets"].format(r=r,b=b,w=w)
        else: answer=t["runway"].format(r=r,b=b,w=w)
        return {"answer":answer,"relationship_id":"current_state_context","language":language}
    if intent == "compare_scenarios":
        first,second=context["scenarios"]; a_name=first["name"]; b_name=second["name"]
        a=first.get("after_state"); b=second.get("after_state")
        # Aggregate terminal scenarios do not identify a ball-by-ball finish or
        # all-out outcome. Do not attempt a rate comparison in that case.
        if a is None or b is None:
            answer={"English":"At least one aggregate scenario reaches a terminal or ambiguous state. Its exact finishing ball or terminal score cannot be inferred from an over summary.","Hinglish":"Kam se kam ek aggregate scenario terminal ya ambiguous hai. Sirf over total se exact finishing ball ya final score nahi nikal sakte.","Hindi":"कम-से-कम एक कुल-ओवर परिदृश्य समाप्ति या अस्पष्ट स्थिति तक पहुँचता है। केवल ओवर के कुल से अंतिम गेंद या अंतिम स्कोर तय नहीं किया जा सकता।","Telugu":"కనీసం ఒక ఓవర్-సారాంశ పరిస్థితి ముగింపు లేదా అనిశ్చిత స్థితికి చేరుతుంది. మొత్తం పరుగుల ఆధారంగా ఖచ్చితమైన చివరి బంతి లేదా స్కోరును నిర్ణయించలేం.","Tamil":"குறைந்தது ஒரு ஓவர்-சுருக்க நிலை முடிவு அல்லது தெளிவற்ற நிலையை அடைகிறது. ஓவரின் மொத்தத்திலிருந்து சரியான முடிவு பந்தை அறிய முடியாது.","Kannada":"ಕನಿಷ್ಠ ಒಂದು ಓವರ್-ಸಾರಾಂಶ ಪರಿಸ್ಥಿತಿ ಅಂತಿಮ ಅಥವಾ ಅನಿಶ್ಚಿತ ಸ್ಥಿತಿಗೆ ತಲುಪುತ್ತದೆ. ಓವರ್ ಮೊತ್ತದಿಂದ ಅಂತಿಮ ಎಸೆತ ಅಥವಾ ಸ್ಕೋರ್ ತಿಳಿಯಲಾಗದು.","Bengali":"অন্তত একটি ওভার-সারাংশ পরিস্থিতি শেষ বা অনিশ্চিত অবস্থায় পৌঁছেছে। শুধু ওভারের মোট রান থেকে শেষ বল বা চূড়ান্ত স্কোর জানা যায় না."}[language]
            return {"answer":answer,"relationship_id":"scenario_difference","language":language}
        if focus_id == "required_rate_difference" and a.get("required_run_rate") is not None and b.get("required_run_rate") is not None:
            gap=abs(float(b["required_run_rate"])-float(a["required_run_rate"]))
            direct="higher" if b["required_run_rate"]>a["required_run_rate"] else "lower" if b["required_run_rate"]<a["required_run_rate"] else "the same"
            answer=t["rates"].format(a=a_name,b=b_name,ar=_rate(a.get("required_run_rate"),language),br=_rate(b.get("required_run_rate"),language),gap=f"{gap:.2f}",direction=DIRECTION_WORDS[language][direct])
        elif focus_id == "wickets_difference":
            answer=t["wicket_comparison"].format(a=a_name,b=b_name,aw=a["wickets_remaining"],bw=b["wickets_remaining"])
        else:
            # Direction must be computed from the ordered pair. A reversed pair
            # must not claim the second scenario needs fewer runs when it needs more.
            delta = int(a["runs_required"]) - int(b["runs_required"])
            if language == "English":
                if delta > 0: comparison = f"That is {delta} fewer runs needed under the second scenario."
                elif delta < 0: comparison = f"That is {abs(delta)} more runs needed under the second scenario."
                else: comparison = "Both scenarios leave the same runs required."
                answer = f"{a_name} leaves {a['runs_required']} runs needed; {b_name} leaves {b['runs_required']}. {comparison} Both are six-legal-ball hypotheticals, not forecasts."
            else:
                prefix_templates = {
                    "Hinglish": "{a} ke baad {ar} runs chahiye; {b} ke baad {br}.",
                    "Hindi": "{a} के बाद {ar} रन चाहिए; {b} के बाद {br} रन चाहिए।",
                    "Telugu": "{a} తర్వాత {ar} పరుగులు కావాలి; {b} తర్వాత {br} పరుగులు కావాలి.",
                    "Tamil": "{a} பிறகு {ar} ரன்கள் தேவை; {b} பிறகு {br} ரன்கள் தேவை.",
                    "Kannada": "{a} ನಂತರ {ar} ರನ್ ಬೇಕು; {b} ನಂತರ {br} ರನ್ ಬೇಕು.",
                    "Bengali": "{a} এর পরে {ar} রান দরকার; {b} এর পরে {br} রান দরকার।",
                }
                comparisons = {
                    "Hinglish": {"fewer": "Second scenario mein {d} runs kam chahiye.", "more": "Second scenario mein {d} runs zyada chahiye.", "same": "Dono scenarios mein same runs chahiye."},
                    "Hindi": {"fewer": "दूसरे परिदृश्य में {d} रन कम चाहिए।", "more": "दूसरे परिदृश्य में {d} रन अधिक चाहिए।", "same": "दोनों परिदृश्यों में समान रन चाहिए।"},
                    "Telugu": {"fewer": "రెండో పరిస్థితిలో {d} పరుగులు తక్కువ కావాలి.", "more": "రెండో పరిస్థితిలో {d} పరుగులు ఎక్కువ కావాలి.", "same": "రెండు పరిస్థితుల్లో ఒకేన్ని పరుగులు కావాలి."},
                    "Tamil": {"fewer": "இரண்டாவது சூழலில் {d} ரன்கள் குறைவாகத் தேவை.", "more": "இரண்டாவது சூழலில் {d} ரன்கள் அதிகமாகத் தேவை.", "same": "இரண்டு சூழல்களிலும் ஒரே அளவு ரன்கள் தேவை."},
                    "Kannada": {"fewer": "ಎರಡನೇ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ {d} ಕಡಿಮೆ ರನ್ ಬೇಕು.", "more": "ಎರಡನೇ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ {d} ಹೆಚ್ಚು ರನ್ ಬೇಕು.", "same": "ಎರಡೂ ಪರಿಸ್ಥಿತಿಗಳಲ್ಲಿ ಒಂದೇ ಸಂಖ್ಯೆಯ ರನ್ ಬೇಕು."},
                    "Bengali": {"fewer": "দ্বিতীয় পরিস্থিতিতে {d} রান কম দরকার।", "more": "দ্বিতীয় পরিস্থিতিতে {d} রান বেশি দরকার।", "same": "দুই পরিস্থিতিতেই একই সংখ্যক রান দরকার।"},
                }
                disclaimers = {
                    "Hinglish": "Dono six-legal-ball hypotheticals hain, prediction nahi.",
                    "Hindi": "ये छह वैध गेंदों की काल्पनिक स्थितियाँ हैं, भविष्यवाणी नहीं।",
                    "Telugu": "ఇవి ఆరు లీగల్ బంతుల కల్పిత పరిస్థితులు, అంచనాలు కావు.",
                    "Tamil": "இவை ஆறு சட்டபூர்வ பந்துகளின் கற்பனை நிலைகள்; கணிப்பு அல்ல.",
                    "Kannada": "ಇವು ಆರು ಕಾನೂನುಬದ್ಧ ಎಸೆತಗಳ ಕಲ್ಪಿತ ಸ್ಥಿತಿಗಳು; ಭವಿಷ್ಯವಾಣಿ ಅಲ್ಲ.",
                    "Bengali": "এগুলি ছয়টি বৈধ বলের কাল্পনিক হিসাব, পূর্বাভাস নয়।",
                }
                kind = "fewer" if delta > 0 else "more" if delta < 0 else "same"
                prefix = prefix_templates[language].format(a=a_name, b=b_name, ar=a["runs_required"], br=b["runs_required"])
                tail = comparisons[language][kind].format(d=abs(delta))
                answer = f"{prefix} {tail} {disclaimers[language]}"
        return {"answer":answer,"relationship_id":"scenario_difference","language":language}
    card=context.get("card",{}); before=card.get("before_state",{}); after=card.get("after_state",{}); claims=context.get("claims",{})
    if intent == "explain_chase_change" and focus_id == "target_reached":
        answer=t["target"].format(r=claims.get("before_runs_required"),b=claims.get("before_balls_remaining"),added=claims.get("runs_added"),score=claims.get("after_score"),target=claims.get("target"),deliveries=claims.get("recorded_deliveries"))
        return {"answer":answer,"relationship_id":"target_reached","language":language}
    values={"bw":before.get("wickets_lost"),"aw":after.get("wickets_lost"),"br":_rate(before.get("required_run_rate"),language),"ar":_rate(after.get("required_run_rate"),language),"bruns":before.get("runs_required"),"bballs":before.get("balls_remaining"),"aruns":after.get("runs_required"),"aballs":after.get("balls_remaining")}
    if focus_id == "wicket_rate_shift": answer=t["wicket_shift"].format(**values); relationship="wickets_and_rate"
    elif focus_id == "rate_shift": answer=t["rate_shift"].format(**values); relationship="rate_changed"
    else: answer=t["before_after"].format(**values); relationship="current_state_context"
    return {"answer":answer,"relationship_id":relationship,"language":language}
