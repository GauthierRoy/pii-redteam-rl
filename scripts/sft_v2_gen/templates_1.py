"""Template bank part 1: support ticket, email with signature, forwarded email
thread, meeting notes. Slot {N} = display name (title baked in for surname
start/middle by the engine). Cores target ~24/37/52 words by bucket order
([20,30] / [31,45] / [46,60]); engine pads to 25/38/53 and verifies."""

TPL = {}

# ---------------- support ticket ----------------
TPL[("support ticket", "start")] = [
    "{N} logged ticket {T} Monday over repeated broadband dropouts every evening after six. Engineers replaced the router, tested the line twice, yet speeds still collapse during peak hours. The provider promised overnight monitoring and a full diagnostic report.",
    "{N} raised ticket {T} Friday after the intercom failed three nights running, leaving visitors stranded outside. Caretakers reset the panel twice and ordered a replacement handset. Management promised same-week repair and temporary door codes.",
    "{N} opened ticket {T} Tuesday over rats spotted near the bin sheds three nights running. Droppings found by the garages, holes dug under fences. The council promised bait boxes, sealed lids, and weekly checks until clear.",
    "{N} filed ticket {T} Wednesday after the stairwell lights failed three nights straight. Residents stumbled on steps, parcels left in darkness. Electricians rewired the circuit Thursday and replaced sensors throughout.",
]
TPL[("support ticket", "middle")] = [
    "The busy housing office quietly reopened long-standing ticket {T} late Friday again after {N} reported mould spreading in the bathroom once more. Bleach treatments failed twice and the extractor hums loudly. Surveyors booked Thursday to inspect ventilation fully.",
    "The leisure desk escalated ticket {T} after {N} complained the pool showers ran cold all week. Boilers checked twice and timers reset. Engineers promise hot water by Friday morning without fail.",
    "The library system locked ticket {T} yesterday after {N} tried renewing borrowed books online Sunday evening. The account shows fines incorrectly, blocking further loans. Staff promised a manual reset Monday morning.",
    "The council helpline reopened ticket {T} late Monday after {N} flagged the missed garden waste pickup twice running. Bins overflowed for weeks and foxes tore bags nightly. Crews promised a catch-up round Friday morning.",
]
TPL[("support ticket", "end")] = [
    "Ticket {T} concerns repeated missed bin collections on our street for three weeks running. Crews left full boxes behind twice despite online reports. Waste now spills across pavements, attracting foxes nightly. Please escalate immediately to area manager {N}\nAccounts, ext. {E}",
    "Ticket {T} queries the sudden closure of the community fridge without notice last Monday. Volunteers found it unplugged and food spoiled. A replacement was promised within days with locks fitted. Please confirm the reopening date for {N}\nDesk, ref. {E}",
    "Ticket {T} disputes the cancelled bulky waste pickup missed twice without warning. Sofa still kerbside and neighbours complain. Crew blamed traffic on both occasions. New slot must be confirmed promptly with {N}\nTeam, bay {E}",
    "Ticket {T} remains open: parcel never arrived despite two delivery attempts last week. Depot claims no access, but the doorbell works fine. Escalate urgently to supervisor {N}\nOffice, ext. {E}",
]
TPL[("support ticket", "signature")] = [
    "Could the team please confirm that ticket {T} was closed correctly last week? I never received the final summary by message. The portal still shows it pending today.\nThanks,\n{N}",
    "Ticket {T} still shows the parcel as delayed despite delivery confirmation photos Monday. The courier claims a safe drop, but nothing arrived at all. Please investigate urgently and update the log.\nThanks,\n{N}",
    "Street light out again outside number twelve, corner pitch dark nightly. Please fix urgently before someone trips badly on the path. School route uses this corner daily.\nThanks,\n{N}",
    "Bin crews missed our close twice running despite online reports filed. Rubbish piled high and gulls scatter it daily. Smell dreadful in this heat wave.\nKind regards,\n{N}",
]

# ---------------- email with signature ----------------
TPL[("email with signature", "start")] = [
    "Subject: pothole damage\n{N} reported suspension damage from the deep pothole outside number twelve Monday. The council logged the claim Tuesday and inspectors photographed the hole Wednesday. Repairs scheduled this week with compensation forms enclosed.",
    "Subject: collapsed drain\n{N} reported the collapsed drain flooding garages Tuesday night. Water knee-deep and cars moved quickly. The council sent pumps Wednesday and engineers survey Thursday. Repairs start next week with the road closed.",
    "Subject: grant outcome\n{N} shared the grant outcome Tuesday with joy: five thousand pounds for hall repairs. Roof first, then wiring. Builders tender shortly and work starts in spring. Celebration tea planned for donors.",
    "Subject: broken lift\n{N} reported the lift breakdown Monday morning, with residents stuck on upper floors. Engineers attended by noon and restored service after replacing a faulty relay switch unit.",
]
TPL[("email with signature", "middle")] = [
    "Subject: noisy radiator\nThe bedroom radiator clanks all night despite bleeding twice, and {N} asked maintenance twice already. A replacement valve was promised last month, still waiting. Please chase the contractor again this week.",
    "Subject: cellar damp\nThe cellar walls weep every winter despite sealing twice, and {N} called the surveyor again Monday. A sump pump was suggested last year and quotes awaited since spring. Decision needed soon.",
    "Subject: slipped tiles\nTwo roof tiles slipped in last week's gales, and young {N} photographed the damage from the garden. The roofer quoted Friday, scaffolding included, with repairs due before further rain arrives.",
    "Subject: skipped bins\nTwo streets missed Friday collections again, and {N} photographed the overflowing bins neatly. Crews return Monday morning with a second lorry. Complaint logged with the depot.",
]
TPL[("email with signature", "end")] = [
    "Subject: parking permits\nThe new permit scheme starts Monday with zone markings repainted over the weekend. Visitor bays move to the far end, disabled spots stay near the entrance. Residents must display badges clearly and report misuse promptly to {N}\nPermits office, ext. {E}",
    "Subject: allotment subs\nPlot rents rise modestly from April with water charges folded in. Invoices posted this week, payment due first May. Please query discrepancies promptly with {N}\nTreasurer, plot {E}",
    "Subject: parish hall invoice\nThe hall repair invoice arrived with duplicate labour charges for painting completed last month. The treasurer queried each line carefully and requested a corrected statement before payment for {N}\nAccounts, ext. {E}",
    "Subject: wasp nest\nA wasp nest grows daily above the shed door, activity rising. Exterminators booked Friday morning and the path will close briefly. Confirm access arrangements with {N}\nPest line, ext. {E}",
    "Subject: icy steps\nFront steps freeze nightly without gritting. Caretaker spreads salt daily for {N}\nStores, bin {E}",
    "Subject: dim landing\nLanding bulb glows weakly despite replacement. Electrician attends Friday for {N}\nRepairs, ref. {E}",
]
TPL[("email with signature", "signature")] = [
    "Could residents please keep the stairwell clear of bikes and prams this week? Fire inspection due Friday morning and access needed throughout. Prams fold flat against the wall.\nThanks,\n{N}",
    "Missed bin lorries left our street strewn with rubbish again Tuesday. Please send a crew before foxes scatter it further across gardens. Recycling boxes were spared this time.\nBest,\n{N}",
    "Playground litter worsens weekly with fast-food wrappers everywhere lately. Please add a bin near the swings before holidays. Foxes scatter bags nightly at present.\nThanks,\n{N}",
    "Potholes crater our lane badly after frosts and cars swerve dangerously. Please inspect urgently this week before school term starts. Cyclists wobble into traffic.\nBest regards,\n{N}",
    "Hedge cuttings block the alley behind garages badly. Please clear them this week before collection.\nThanks,\n{N}",
    "Dog fouling mars the playing field daily now. Please bag it and bin it every visit.\nRegards,\n{N}",
    "Fly-tipping scars the lane end again lately. Please report plates promptly to the team.\nThanks,\n{N}",
    "Mud slicks the towpath badly after rain weekly. Please grit the worst stretch soon.\nBest,\n{N}",
]

# ---------------- forwarded email thread ----------------
TPL[("forwarded email thread", "start")] = [
    "Fwd: shed break-in\n{N} reported the allotment shed break-in Sunday night with tools stolen and the padlock cut. Police logged the crime number Monday morning and neighbours shared camera footage. The committee will replace locks and review insurance.",
    "Fwd: street party\n{N} booked the bouncy castle for the street party Saturday with deposit paid. Road closure approved and notices posted. Raffle tickets printed and the cake rota agreed. Face painting confirmed for afternoon.",
    "Fwd: gas readings\n{N} sent the quarterly gas readings Tuesday with meter photos attached. Bills expected lower this quarter after mild weather. Direct debit stays unchanged for now.",
    "Fwd: hall keys\n{N} lost the spare hall keys Sunday after the concert chaos. Locks changed Monday morning and a new set cut. Please collect from the office with ID and sign the log.",
]
TPL[("forwarded email thread", "middle")] = [
    "Fwd: traffic calming\nLong thread below on speeding past the school gates each morning lately. Parents logged plates twice and police visited once. {N} collated the counts into charts for the council meeting. A crossing patrol is requested urgently.",
    "Fwd: flood alerts\nLengthy thread on blocked culverts behind the allotments again this month. Water pools after rain and sheds flood badly. {N} filmed the overflow Tuesday and sent stills to the drainage board. Jetting requested urgently.",
    "Fwd: verge cutting complaint\nQuick note following Thursday's near miss at the junction: {N} asked the council to cut the blind corner verge again urgently. Visibility dreadful and mirrors clipped twice already. Gang booked for Monday morning.",
    "Fwd: pigeon mess\nEndless thread on pigeons fouling the balconies daily this summer. Nets proposed and spikes quoted twice. {N} tested deterrent gel Sunday with mixed results so far. Committee votes Tuesday evening.",
]
TPL[("forwarded email thread", "end")] = [
    "Fwd: damp survey\nThe surveyor found rising damp in two ground-floor rooms with plaster blown throughout. Quotes invited from three firms, work due before winter. Report filed by {N}\nSurvey ref. {E}",
    "Fwd: bench stain\nThe memorial bench needs restaining before winter sets in properly. Volunteers meet Saturday morning with brushes and sandpaper. Paint ordered by {N}\nStores, shelf {E}",
    "Fwd: pool rota gaps\nLifeguard cover runs thin Sunday with two off sick at once. Volunteers needed urgently all weekend for races. Shifts confirmed by {N}\nLeisure desk, ext. {E}",
    "Fwd: gate code\nThe side gate code changed Sunday after the lock jammed twice. New number issued to residents only and memo pinned. Details confirmed by {N}\nOffice, ref. {E}",
    "Fwd: salt run\nQuick note Friday: {N} gritted the steep steps twice. Bin refilled, shovel stored dry.",
    "Fwd: hedge trim\nHeads up Monday: {N} cut back the alley hawthorn hard. Waste bagged, path swept clean.",
    "Fwd: bulb order\nSpring bulbs need ordering soon for the verges. Suggestions to the meeting please tonight. Tulips popular last year with {N}",
]
TPL[("forwarded email thread", "signature")] = [
    "Fwd: hall bookings\nThe December concert booking clashes with the youth club party planned months ago. Both groups paid deposits and neither will move. Caretakers offered Sunday instead with heating included. Please confirm acceptance soon.\nBest regards,\n{N}",
    "Fwd: fete accounts\nThe summer fete accounts balance at last with every receipt matched. Takings banked Tuesday and floats returned. Suppliers paid in full and audit signed off cleanly.\nBest regards,\n{N}",
    "Fwd: coach trip\nThe seaside coach trip needs six more names by Friday or it cancels outright. Seats cost twelve pounds, lunch included. Deposits to the office by Wednesday latest.\nBest,\n{N}",
    "Fwd: insurance renewal\nThe hall insurance renewal jumped thirty percent with no claim made all year. Brokers blame market rates and trustees demand options. Three quotes requested and cover checked.\nKind regards,\n{N}",
]

# ---------------- meeting notes ----------------
TPL[("meeting notes", "start")] = [
    "{N} opened the flood defence review by summarising river gauge readings from the past fortnight. Councillors debated sandbag stores, pump maintenance schedules, and warning siren tests. Volunteers were assigned to check vulnerable cottages.",
    "{N} presented the playground inspection findings Thursday with photos of worn safety surfacing. Councillors agreed urgent repairs and quotes invited from two firms. Work starts at half term, funded from reserves.",
    "{N} flagged the faulty gate lock Monday and asked for a locksmith quote. Governors approved the cost Tuesday and work booked for Friday morning. Spare keys cut for wardens.",
    "{N} tabled the parking survey results Tuesday showing overwhelming support for permits. Councillors debated zones, charges, and visitor rules. A trial starts in spring covering three streets.",
]
TPL[("meeting notes", "middle")] = [
    "The allotment committee reviewed water charges Thursday and voted to install separate meters per row. Taps leaked all summer and bills doubled, so {N} proposed monthly readings with a volunteer rota. Plot holders agreed unanimously.",
    "School fete planning started Monday with stalls allocated and floats ordered. {N} offered the sound system for the afternoon, cables tested beforehand. Raffle books printed and posters pinned to boards.",
    "Village hall committee reviewed hiring fees Tuesday and agreed a small increase from April. {N} opposed the rise, citing pensioners on fixed incomes. A compromise freezes weekday mornings instead.",
    "Allotment AGM set for March with subscriptions unchanged this year. {N} volunteered for the audit team, seconded unanimously amid applause. Date for the spring show fixed afterwards.",
]
TPL[("meeting notes", "end")] = [
    "Playground surface repairs approved Tuesday evening, contractor starts next Monday morning sharp without fail. Safety mats ordered, fence posts replaced. Final inspection walkthrough booked with caretaker {N}",
    "Christmas market stalls allocated Friday with power points mapped and tested. Food traders placed near the entrance, crafts along the far wall. Thanks recorded to all volunteers and organiser {N}",
    "Skittles evening booked Friday with teams drawn from all streets. Bar rota agreed, snacks ordered, scoresheets printed. Trophy engraving ordered for captain {N}",
    "Darts final arranged Saturday with buffet ordered and barrel tapped early. Draw made fairly and handicaps set by committee. Presentation follows play with trophy for {N}",
]
TPL[("meeting notes", "signature")] = [
    "Blocked gutters were cleared Saturday with two heavy bags of moss removed. All downpipes run free again now and rodding stored.\nThanks,\n{N}",
    "Church roof fund passed two thousand pounds Sunday with gift aid still to claim. Donors thanked warmly and list pinned to the board.\nBest regards,\n{N}",
    "Grit bins refilled Friday at all hilltop spots before the frost. Volunteers thanked warmly and new rota pinned to the board.\nCheers,\n{N}",
    "Village show planning launched Monday with classes set and judges booked. Tents ordered and tables hired. Schedules printed, entries open.\nBest regards,\n{N}",
]
