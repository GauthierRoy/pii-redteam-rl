"""Template bank part 2: chat, appointment reminder, form/record, list/table,
informal typos, third-person report. Slot {N}. Cores ~24/37/52 words."""

TPL = {}

# ---------------- chat message ----------------
TPL[("chat message", "start")] = [
    "{N} here, found a set of keys by the bike sheds this morning. Audi fob plus two Yale keys on a red ring. Describe them to claim, I am in flat twenty-two most evenings.",
    "{N} here: the intercom keeps buzzing every flat at midnight again this week. Caretaker reset the panel twice Tuesday, still faulty. Please log an urgent repair before the weekend rush.",
    "{N} left a parcel for you with flat six downstairs. Collect after five please, knock twice loudly as the bell is broken again today.",
    "{N}: bus stop glass smashed again last night opposite the shops. Council informed this morning and cleanup requested urgently before the school run starts.",
]
TPL[("chat message", "middle")] = [
    "Hey, can someone cover my Saturday morning shift at the cafe? The rota shows {N} down for till duty from eight, but the bus drivers are striking again. I can swap Sunday afternoon instead, just confirm here soon.",
    "Anyone free to water the greenhouse this weekend? The rota has {N} down for Saturday morning, but the taps need a new washer first. Tools are in the shed corner.",
    "Late home tonight, train cancelled again with no replacement bus. Borrowed a charger from {N} at the station, total lifesaver. Should reach by ten if traffic clears.",
    "Evening shift swap needed Saturday as the twins have a swimming gala all day. The rota lists {N} on till duty from noon, but school runs clash badly. I can cover Friday evening instead.",
]
TPL[("chat message", "end")] = [
    "Anyone seen my blue umbrella? Left it in the community hall after yoga class last night. Checked lost property twice already with no luck. If found, please leave it with reception for {N}",
    "Does anyone have the spare key for the bike shed? Mine snapped in the lock yesterday evening and the caretaker is off sick. If you can help, please knock loudly at flat twelve for {N}",
    "Lost: grey tabby cat, white paws, missing since Sunday night near the allotments. Very timid with strangers, please check sheds and garages. Reward offered, contact details with {N}",
    "Has anyone seen the missing parcel for flat nine? Tracking shows delivered Thursday but nothing arrived. The driver left no card and neighbours saw nothing. Please check cupboards and sheds for {N}",
]
TPL[("chat message", "signature")] = [
    "Book club moved to Thursdays at seven in the back room, new members always welcome with wine. This month we finish the dystopia novel, next up poetry selections. Reminders go out Sunday evening.\ncheers\n{N}",
    "Spare parking permit going spare for zone A, valid till March. First to collect from flat nine gets it, no holding back now.\nThanks,\n{N}",
    "Found a silver bracelet by the swings Sunday, describe it to claim. Ask at flat three most evenings this week.\nThanks,\n{N}",
    "Quiz team short one player Thursday, pub league decider looming. Meet at seven sharp, questions start half past. Bring lucky pens and cash for the kitty.\ncheers\n{N}",
]

# ---------------- appointment reminder ----------------
TPL[("appointment reminder", "start")] = [
    "{N}, your boiler service is booked Friday morning between nine and noon with the contracted engineer. Please clear the airing cupboard, unlock the side gate, and keep pets indoors. Bring last year's certificate if kept.",
    "{N}, your car MOT retest is booked Saturday morning at half past eight in lane three. Please bring last week's failure sheet, service history, and photo ID. Arrive ten minutes early and park in bay five.",
    "{N}, your annual boiler check is due next month with servicing slots open weekdays. Please pick a morning that suits and confirm by phone. Clear the cupboard beforehand and leave parking space outside.",
    "{N}, your dentist appointment is moved to Monday morning at half past nine in surgery two. Please arrive early with your medical card and current prescriptions. Inform reception of any changes promptly.",
]
TPL[("appointment reminder", "middle")] = [
    "Your annual boiler service visit moves to next Thursday morning because {N} double-booked the early round. Please leave the key with a neighbour and clear the cupboard fully. Engineers call ahead on arrival day.",
    "Your flu clinic slot moves to Friday morning because {N} double-booked the early session. Please arrive ten minutes early and wear short sleeves. No booking needed for family members.",
    "Your window cleaning slot shifts to Wednesday morning because {N} rearranged the whole round after van repairs. Please leave gates unlocked and move cars beforehand. Ladders need firm dry ground.",
    "Your boiler service moves to Thursday because {N} double-booked the morning round after leave changes. Please leave the key with a neighbour and clear the cupboard. Engineers call ahead on arrival.",
]
TPL[("appointment reminder", "end")] = [
    "Relaxed Saturday flu jabs session at the village hall, walk-ins welcome all morning long. Two family slots held and booked for {N}",
    "Your boiler service is scheduled Friday between nine and noon. Please clear access to the airing cupboard and leave the key with {N}",
    "Your eye test is booked Monday at half past two in clinic room six. Please bring current glasses and a list of medicines. Ask for {N}",
    "Evening harvest supper tickets ready for collection Friday at the hall office. Tables seat eight, doors open half past six sharp. Gluten-free options booked for {N}",
]
TPL[("appointment reminder", "signature")] = [
    "Your dental checkup is booked Thursday at half past ten with the hygienist first. Please arrive ten minutes early and bring your medical card. Reply to confirm or reschedule promptly.\nBest,\n{N}",
    "Your smear test is due this month and lasts ten minutes with a nurse. Morning slots available Tuesday and Thursday weekly. Please book online or call reception to arrange.\nSincerely,\n{N}",
    "Routine gas safety checks run next week across all flats with access needed daily. Engineers carry ID and wear overshoes. Please clear meters, confine pets, and note faults.\nKind regards,\n{N}",
    "Flu jabs shifted to Thursday morning with extra nurses drafted in. Bring cards, wear short sleeves, and queue outside calmly.\nThanks,\n{N}",
]

# ---------------- form/record line ----------------
TPL[("form/record line", "start")] = [
    "{N} \u2014 allotment transfer approved with shed, greenhouse, keys. Fees paid, rules signed. Plots measured twice and soil tested yearly. Waiting list updated, induction booked Friday.",
    "{N} \u2014 skip hire booked for Saturday morning, access via side gate. Fill only with soil and rubble, no plasterboard allowed. Permit displayed in the front window.",
    "{N} \u2014 boundary fence dispute logged with photos, survey maps, and witness statements. Hedge height measured twice, deeds checked at the registry. Mediation booked for October.",
    "{N} \u2014 meter reading disputed for March quarter, estimated bill far above actual usage shown. Photos submitted, corrected invoice requested, direct debit paused pending review.",
]
TPL[("form/record line", "middle")] = [
    "Planning file 25/118 \u2014 rear dormer refused Tuesday over height and bulk. Drawings revised twice already, yet officers still object, so {N} lodged a formal appeal yesterday. Hearing date awaited, barrister instructed.",
    "Tree works order raised Monday: oak crown lifted, deadwood removed safely. {N} supervises Friday morning, chipper booked, logs stacked neatly. Neighbours notified by letter.",
    "Skip hire docket 44 \u2014 soil only, weighed in at the gate. {N} booked Saturday collection, access arranged beforehand. Permit displayed in cab window, overloads charged extra.",
    "Tenant file Z-118 \u2014 arrears cleared in full August, ledger balanced. Deposit protected and inspection passed. {N} listed as occupant since spring with rent direct debited monthly.",
]
TPL[("form/record line", "end")] = [
    "Ref: LM-2291 \u2014 annual boiler service visit completed Thursday afternoon by the contracted engineer, filter replaced, pressure tested, radiators bled. Certificate issued and filed for {N}",
    "Ref: HW-0712 \u2014 November damp proofing invoice queried line by line, two labour lines duplicated, materials overcharged. Revised statement requested, payment withheld until corrected by {N}",
    "Ref: CM-3300 \u2014 November memorial bench plaque wording approved unanimously, dates checked twice, spelling confirmed against records. Foundry casting booked for {N}",
    "Ref: WS-5510 \u2014 autumn hedge cutting invoice approved after inspection, hours verified against the log, waste removed tidily. Risk forms filed duly. Payment released to contractor {N}",
    "Ref: BK-12 \u2014 lost library book repaid Tuesday, fine cleared in full. Account settled for {N}",
    "Ref: PT-08 \u2014 hall deposit refunded Friday, keys returned duly. Booking closed for {N}",
]
TPL[("form/record line", "signature")] = [
    "Hall heating repair logged Thursday morning, replacement parts ordered, invoice pending approval by the treasurer after the site visit.\nSigned\n{N}",
    "Quarterly allotment fees collected in full last weekend, receipts issued promptly to all plot holders, arrears chased by letter twice. Waiting list updated after inspection.\nThanks,\n{N}",
    "Lost property auction proceeds banked Friday morning, unclaimed items donated to charity shops locally after notices expired recently.\nSigned\n{N}",
    "Playground grant spent in full with invoices filed, receipts matched, photos archived. New swings installed, slide resurfaced, fences repainted. Safety certificate issued Monday.\nRegards,\n{N}",
]

# ---------------- list/table row ----------------
TPL[("list/table row", "start")] = [
    "{N} | boiler flue swept | condensate trap cleared | pressure logged | certificate filed | next service spring | invoice settled | warranty stamped | radiators bled",
    "{N} | hall deep clean booked | floors scrubbed | windows washed | bins emptied | keys returned | toilets stocked | stage swept | curtains steamed",
    "{N} | cake stall Saturday | buns priced | float counted | change ready | napkins folded | tea urn on | cups stacked | trays washed",
    "{N} | gutter clearance round booked | ladders checked | bags loaded | downpipes flushed | moss scraped | joints sealed | brackets tightened | waste bagged",
]
TPL[("list/table row", "middle")] = [
    "spring bulb order placed | tulips two sacks | daffodils one sack | {N} | compost topped up | beds weeded | labels written | watering rota set | tools oiled",
    "allotment seed order ledger | compost delivered Friday | tools sharpened | {N} | invoices checked twice | plots measured | water rota pinned | AGM date set",
    "fundraising ledger update | gift aid claimed | hall hire settled | {N} | arrears chased twice | receipts filed | audit due June | floats counted",
    "tool hire log | ladders returned | drill booked | {N} | deposit refunded | late fee waived | next slot Tuesday | ID checked",
]
TPL[("list/table row", "end")] = [
    "winter gritting rota agreed | pavements salted nightly | bus stops prioritised | schools alerted early | supplies restocked weekly | salt barn full | volunteers thanked | logged by {N}",
    "pothole survey results mapped | drains jetted | verges cut | signs cleaned | lines repainted | gullies emptied | bush cut back | logged for {N}",
    "winter shelter rota published | blankets collected | soup urn ordered | volunteers vetted | hall heated nightly | doors open seven | coats racked | referrals logged | thanked warmly by {N}",
    "churchyard tidy rota agreed | graves weeded | paths swept | bins emptied | tap fixed | mower serviced | hedge trimmed | bulbs planted | records updated | logged with thanks by {N}",
    "pantry stocktake done | tins dated | jars sealed | packets closed | logged by {N}",
    "bulb boxes stacked | labels faced | twine cut | tags tied | counted by {N}",
    "gate roster pinned | keys signed out | lamps checked | paths walked | covered by {N}",
    "toy library list updated | pieces counted | boxes latched | cards stamped | run by {N}",
    "pond rota agreed | weed cleared | litter netted | banks strimmed | watched by {N}",
    "book sale float ready | change bagged | prices stuck | tables set | banked by {N}",
]
TPL[("list/table row", "signature")] = [
    "autumn fair takings counted twice | stalls cleared by six | lost property boxed | raffle profit banked | thanks recorded | cakes sold out | -- {N}",
    "bonfire wood collected | pallets stacked | embers damped | ashes scattered | site left tidy | thanks all | -- {N}",
    "jumble sale proceeds banked | clothes rail cleared | books boxed | toys sorted | takings counted twice | thanks minuted | cakes sold out | -- {N}",
    "harvest auction lots catalogued | apples pressed | honey jarred | pumpkins weighed | bread baked dawn | crafts priced | music booked | thanks sung loudly | -- {N}",
]

# ---------------- informal text with typos ----------------
TPL[("informal text with typos", "start")] = [
    "{N} dont forget teh wellies tomorow, field soaked agian! I loaded the car with flasks, snacks, and spare socks last night. The match kicks off at ten sharp regardless.",
    "{N} rember swimming kits tomorow, goggles too! Pool closed last week, reopened today luckily. Lessons restart Monday and groups stay the same. Dont be late again.",
    "{N} rember the picnic blankets tomorow morning! I packed sandwiches, fruit, and juice cartons, dont forget the cups, plates, and napkins please.",
    "{N}: tap dripping agian in flat four bathroom all night. Washer gone completely, plumber needed soon before teh bill climbs steeply.",
]
TPL[("informal text with typos", "middle")] = [
    "Sorry mate, forgot to charge the power bank agian last night, totally my fault so dont worry. The spare cable sits in my backpack side pocket, so old {N} can still plug in at the cafe later. We meet around noon tomorrow.",
    "Doctor running late agian today, waiting room packed with coughing kids. The nurse says {N} must fast overnight, dont eat after midnight please bring the old prescription list and insurance card.",
    "Train delayed agian tonight, stuck outside the station with no announcements. The guard says {N} should claim refunds online, dont bother queueing tomorrow morning.",
    "Washer flooded teh kitchen floor agian last night, water everywhere. The landlord says {N} must claim on insurance, dont delay the paperwork or pay excess. Mop-up took hours.",
]
TPL[("informal text with typos", "end")] = [
    "Bus is runing twenty minutes late agian, stuck behind roadworks near the market. Save me a seat at the back near the window please. Dont let the driver leave without {N}",
    "Kettle packed up this morning, no hot drinks agian. Teh shop sells cheap ones for fifteen pounds, will check after work with {N}",
    "Phone screen cracked agian after dropping it on teh pavement. Repair shop quotes eighty pounds, takes two days. Will ask {N}",
    "Finally got the garden fence fixed yesterday after weeks of nagging the landlord. Painted both sides green, replaced three rotten posts, swept all teh mud away. Will surely ask advice from {N}",
]
TPL[("informal text with typos", "signature")] = [
    "Teh washing machine keeps leaking all over the kitchen floor agian, nightmare. Mopped twice today already, towel supply gone.\ncheers\n{N}",
    "Shower leaking agian all over teh bathroom floor since Monday morning, tiles ruined. Plumber coming Thursday hopefully, buckets out for now.\ncheers\n{N}",
    "Oven wont light at all, gas smells faintly near the hob, getting worried now. Engineer phoned twice today, no answer yet, will try agian tomorow.\nBest,\n{N}",
    "Central heating rattles loudly all night and keeps waking teh baby agian. The engineer blames airlocks and promises parts soon. Nights are exhausting.\nThanks,\n{N}",
]

# ---------------- third-person report ----------------
TPL[("third-person report", "start")] = [
    "{N} opened the flood defence review by summarising river gauge readings from the past fortnight. Councillors debated sandbag stores, pump maintenance schedules, and warning siren tests. Volunteers were assigned to check cottages.",
    "{N} chaired the flood meeting firmly Tuesday and approved sandbag stores. Volunteers listed and rota pinned to boards. River levels checked nightly and alerts tested regularly.",
    "{N} swept the village pond clean Sunday with volunteers hauling weed all morning. Ducks returned by evening and moorhens nested again. The council funds new reeds next spring.",
    "{N} mended the broken stile Sunday with spare planks and nails. Walkers thanked the quick repair warmly on social media. Path reopened Monday morning for all.",
]
TPL[("third-person report", "middle")] = [
    "The parish council debated the noisy late-night market for two hours Tuesday evening. Stallholders demanded longer hours while residents complained bitterly, so {N} proposed a trial compromise with earlier closing times. A vote is scheduled next month.",
    "The allotment society praised the new compost bays Sunday after months of building work. Bays filled evenly and rotas posted, so {N} donated tools worth hundreds and trained beginners patiently. Membership rose sharply afterwards.",
    "Village hall committee reviewed hiring fees Tuesday and agreed a small increase from April. {N} opposed the rise, citing pensioners on fixed incomes. A compromise freezes weekday mornings instead.",
    "The darts league welcomed new teams Friday with handicaps set fairly. {N} drew the fixtures live and posted results online. Play starts Monday evening with boards checked.",
]
TPL[("third-person report", "end")] = [
    "The science fair crowned its young winners Saturday amid cheering families. Robots raced fiercely and rockets soared highest. The headteacher closed the day praising star pupil {N}",
    "The parish newsletter praised the revamped playground warmly with photos of delighted toddlers. Swings repainted, slide resurfaced, benches varnished. Toddlers queued daily, all thanks to fundraiser {N}",
    "The music festival closed Sunday with encores ringing out loudly. Bands played late and crowds danced wildly. Takings beat records and stalls sold through. The grand finale honoured director {N}",
    "Remembrance parade routes agreed Sunday with road closures timed to the minute. Bands placed at the front, veterans behind colours. Wreaths ordered and bugler booked. Dismissal points mapped for crowds honouring {N}",
]
TPL[("third-person report", "signature")] = [
    "The summer fete raised record funds Saturday with stalls busy till dusk. Takings counted twice under supervision and prizes awarded fairly. Organisers praised every helper warmly in the closing speech.\nKind regards,\n{N}",
    "The harvest supper drew ninety guests Friday with puddings judged at midnight. Raffle takings beat last year by half. Helpers stayed late washing up cheerfully together.\nBest regards,\n{N}",
    "The bonfire committee thanked crowds Sunday after the biggest turnout in years. Gates ran smoothly and food sold out. Fireworks thrilled all ages and music played late.\nThanks,\n{N}",
    "The knit-and-natter group donated forty blankets Tuesday for newborns. Wool gifted and needles clicking nightly. Mothers thanked the makers tearfully at handover.\nKind regards,\n{N}",
]
