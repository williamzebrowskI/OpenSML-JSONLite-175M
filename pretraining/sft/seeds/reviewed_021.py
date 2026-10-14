"""CC0 paired contracts: instruction changes must change the correct answer.

Fictional source notes are explicitly authored facts, not claims about real people.
Gold text transformations use Python operations; Python code uses literal edge
oracles plus a bounded interpreter. Value variants are correlated, not new skills.
"""
import json
from ..balanced import pair
from ..behavior_v8 import python_checks
from ..quality import digest
from .reviewed_019 import TEXT, PYTHON as P19
from .reviewed_020 import PYTHON as P20

PROVENANCE=dict(repo='local/contrast-instructions-021',revision='contrast-instructions-v1',split='authored',license='CC0-1.0',method='Explicit fictional facts and reviewed instruction contracts; executable target oracles; correlated pairs remain together')

# Distinct content, not a sequence of renamed numbered rooms.
SCENARIOS=[
('archive','Nell','Soren','atlas','journal','reading desk','study bench'),
('pottery','Keira','Tomas','bowl','vase','drying shelf','work table'),
('music','Amina','Victor','flute','cello','practice room','concert hall'),
('hiking','Felix','Dalia','compass','tent','trail office','camp entrance'),
('painting','Celia','Anton','brush','canvas','studio cabinet','easel stand'),
('baking','Ravi','Elsa','whisk','tray','kitchen counter','pantry shelf'),
('sewing','Petra','Idris','thread','fabric','supply basket','cutting table'),
('cycling','Jonas','Leila','helmet','pump','repair station','garage bench'),
('photography','Maren','Ivo','tripod','lens','equipment closet','editing desk'),
('chess','Vera','Kamal','board','clock','club cupboard','registration desk'),
('lettering','Nadia','Oskar','stencil','marker','design table','storage tote'),
('weaving','Hana','Boris','yarn','shuttle','loom bench','thread drawer'),
('carpentry','Evan','Zara','plane','clamp','tool wall','assembly bench'),
('stage','Luca','Mina','script','costume','rehearsal desk','dressing rail'),
('printing','Otto','Farah','ink','paper','print cabinet','packing table'),
('rowing','Lena','Amir','oar','vest','boathouse rack','launch shed'),
('modelmaking','Dora','Jace','glue','ruler','hobby drawer','build mat'),
('ceramics','Anya','Luis','tile','mold','kiln room','mixing bench'),
('embroidery','Tessa','Malik','needle','hoop','craft box','fabric shelf'),
('rehearsal','Greta','Nico','score','stand','music cabinet','auditorium'),
('repair','Rena','Pavel','pliers','wrench','service cart','parts desk'),
('collage','Esra','Finn','scissors','paste','art cupboard','display table'),
('aquarium','Willa','Ciro','net','bucket','supply room','cleaning station'),
('badminton','Mika','Nora','racket','shuttlecock','sports locker','court bench'),
('calligraphy','Adele','Rohan','pen','inkwell','writing desk','drying board'),
('camping','Simone','Basil','lantern','stove','gear shelf','outdoor table'),
('bookbinding','Una','Theo','needle','cover','binding bench','press table'),
('orchestra','Demi','Hugo','baton','folder','conductor desk','library cart'),
('kite','Bruno','Rhea','string','frame','craft shelf','park kiosk'),
('snowshoe','Alina','Rolf','pole','gaiter','rental counter','equipment rack'),
('stationery','Pia','Milo','notebook','envelope','front desk','mail shelf'),
('puppetry','Rosa','Emil','puppet','screen','stage box','backdrop rail'),
('sculpture','Dina','Yuri','chisel','clay','tool cabinet','modeling stand'),
('basketry','Talia','Gavin','reed','handle','material rack','weaving table'),
('fencing','Erin','Sami','foil','mask','club office','safety rack'),
('skating','Helena','Dan','skates','guards','rental shelf','changing bench'),
('telescope','Nila','Simon','eyepiece','mount','observation shed','assembly table'),
('ceremony','Lara','Ari','banner','ribbon','committee room','supply trunk'),
('festival','Mira','Tobin','poster','sign','ticket office','information booth'),
('workshop','Faye','Ren','apron','goggles','entry hook','safety cabinet'),
('catalog','Kira','Davi','label','stamp','sorting station','filing desk'),
('puzzle','Sana','Hector','piece','box','game shelf','club table'),
('origami','Edda','Imran','paper','guide','folding table','book rack'),
('mosaic','Clara','Yasin','tile','grout','workbench','supply cart'),
('diorama','Juno','Leon','base','figurine','project shelf','painting booth'),
('woodturning','Anika','Russ','blank','gouge','stock rack','tool stand'),
('juggling','Nessa','Marco','clubs','balls','practice basket','prop case'),
('crochet','Imogen','Silas','hook','wool','craft pouch','wool basket'),
('rowingclub','Selma','Axel','paddle','rope','dock shed','supply hook'),
('letterpress','Lucia','Kofi','type','roller','composing desk','press shelf'),
('lacrosse','Maisie','Dante','stick','ball','team room','equipment cage'),
('beading','Anita','Clem','beads','wire','sorting tray','work spool'),
('drama','Eliza','Noel','prop','backdrop','stage cupboard','scene shop'),
('quilting','Livia','Omar','batting','pattern','fabric bin','pattern drawer'),
('harmonica','Jaya','Luc','case','cleaner','band shelf','practice desk'),
('engraving','Monika','Alec','plate','burin','print drawer','cutting stand'),
('sailing','Linnea','Rico','sail','cleat','rigging shed','parts locker'),
('boardgame','Ilse','Tariq','tokens','cards','game cabinet','dealer table'),
('mountaineering','Zola','Karl','harness','carabiner','gear counter','climbing rack'),
('tapestry','Avril','Nadir','bobbin','loom','thread shelf','weaving room'),
]

def authored_rows():
    rows=[]
    def add(bucket,group,messages,pair_id=None,**extra):
        rows.append(dict(bucket=bucket,component='python' if bucket=='python' else 'constraints' if bucket=='contrast_text' else 'chat',group=group,messages=messages,pair_id=pair_id,provenance=PROVENANCE,**extra))
    def contrast(bucket,group,key,source,requests,targets,**extra):
        assert len(requests)==len(targets)==2 and targets[0]!=targets[1]
        identity=group+':'+key
        for side,(request,target) in enumerate(zip(requests,targets)):
            add(bucket,group,pair(source+'\n\n'+request,target),identity,contrast_side=side,rules=[dict(kind='exact',args=[target])],**extra)

    for i,(topic,a,b,x,y,p,q) in enumerate(SCENARIOS):
        group='contrast-021:scene-'+topic
        facts=[(a,x,p),(b,y,q)]
        # Three styles/orderings keep the relation, rather than the position,
        # responsible for the answer. Pairs always share the identical source.
        for style in range(3):
            first,second=facts if style!=1 else facts[::-1]
            if style==0:
                source=f'{first[0]} owns the {first[1]} and placed it at the {first[2]}. {second[0]} owns the {second[1]} and placed it at the {second[2]}. Neither person moved the items afterward.'
            elif style==1:
                source=f'Workshop note: the {first[1]} at the {first[2]} belongs to {first[0]}. At the {second[2]} is the {second[1]} owned by {second[0]}.'
            else:
                source=f'Item | Owner | Location\n{x} | {a} | {p}\n{y} | {b} | {q}'
            contrast('contrast_text',group,f'owner-{style}',source,[f'Who owns the {x}? Give only the name.',f'Who owns the {y}? Give only the name.'],[a,b],facts=facts)
            contrast('contrast_text',group,f'item-{style}',source,[f'Which item belongs to {a}? Give only the item.',f'Which item belongs to {b}? Give only the item.'],[x,y],facts=facts)
            contrast('contrast_text',group,f'location-{style}',source,[f'Where is the {x}? Return only the location.',f'Where is the {y}? Return only the location.'],[p,q],facts=facts)
        source=f'Items on this list, in order: {x}, {y}, {topic}.'
        contrast('contrast_text',group,'first-last',source,['Return only the first listed item.','Return only the last listed item.'],[x,topic])
        contrast('contrast_text',group,'bullet-vs-comma',source,['Return all three items as three dash bullet points, without other text.','Return all three items on one line, separated by a comma and a space.'],['\n'.join('- '+z for z in (x,y,topic)),', '.join((x,y,topic))])
        contrast('contrast_text',group,'case',f'Label: {topic}', ['Return the label in uppercase, without other text.','Return the label in lowercase, without other text.'],[topic.upper(),topic])
        contrast('contrast_text',group,'json-vs-value',f'The owner of the {x} is {a}.',[f'Return only a JSON object containing exactly the key "owner" and its value. Use the ordinary JSON string syntax.',f'Return only the owner\'s name, without quotes or a field label.'],[json.dumps({'owner':a}),a])
        absent=['Absent','Unavailable','Not recorded','No information'][i%4]
        contrast('contrast_text',group,'known-vs-absent',f'{a} owns the {x}. No purchase price is recorded.',[f'What does {a} own? Return only the item.',f'What was the purchase price? Return only "{absent}" if it is not recorded.'],[x,absent])
        # Different operations require different final answers from the same
        # initial history. Every intervening reply is supervised and accurate.
        for op in ('remove','replace','append'):
            start=pair(f'My project uses {x}, {y}, and {topic}.',f'Your project list is {x}, {y}, {topic}.')
            if op=='remove':change=f'Take {y} off the project list.';items=[x,topic]
            elif op=='replace':change=f'Use {p} in place of {y}.';items=[x,p,topic]
            else:change=f'Add {p} to the end of the project list.';items=[x,y,topic,p]
            history=start+pair(change,'The current list is '+', '.join(items)+'.')
            identity=group+':history-'+op
            for side,(request,target) in enumerate([(f'Give only the current list, with commas between the items.',', '.join(items)),('Give only the number of items on the current list.',str(len(items)))]):
                add('dialogue',group,history+pair(request,target),identity,contrast_side=side,rules=[dict(kind='exact',args=[target])])
        note=pair(f'I would like to revise a note about the {topic}.', 'Please paste the note and describe the tone you want.')
        supplied=f'{a} will bring the {x}. Please wait at the {p}.'
        identity=group+':missing-versus-supplied'
        add('dialogue',group,note+pair('I have not written the note yet. What should I send?', 'Send the draft when it is ready, along with the intended recipient and tone.'),identity,contrast_side=0,review_contract='Requests absent input; no invented draft.')
        add('dialogue',group,note+pair('Here is the note: '+supplied+' Copy the second sentence only.',f'Please wait at the {p}.'),identity,contrast_side=1,rules=[dict(kind='exact',args=[f'Please wait at the {p}.'])])

    # Pairs contrast producing code with tracing that very same function. The
    # arithmetic/list operation never changes silently between prompt/target.
    forbidden={'count_positive','clamp','lookup','get_default','lengths','offset_values','positives','first_positive','first','last','divisible','head_slice','tail_slice'}
    for family,args,body,meaning,inputs,expected,explanation in P19+P20:
        if family in forbidden:continue  # Fresh probe semantic contracts remain held out.
        group=('reviewed-019:' if any(p[0]==family for p in P19) else 'reviewed-020:')+'python-'+family
        for v in range(12):
            fn=family+'_task'+str(v%3+1)
            code=f'def {fn}({args}):\n'+''.join('    '+line+'\n' for line in body.splitlines()).rstrip()
            answer='```python\n'+code+'\n```'
            tests=[dict(args_literal=repr(a),expected_literal=repr(e)) for a,e in zip(inputs,expected)]
            assert all(python_checks(answer,tests,function=fn).values()),family
            requests=[f'Implement {fn}({args}): {meaning}.',f'I need {fn}({args}). Its job is to {meaning}.',f'Create a function named {fn}, with arguments {args}, that will {meaning}.',f'Python task: {meaning}. Use the signature {fn}({args}).'][v%4]
            identity=group+':code-trace-'+str(v)
            add('python',group,pair(requests+' Return only a Python code block.',answer),identity,contrast_side=0,functional_assertions=tests,python_contract=dict(function=fn,code_only=True,docstring=False),review_contract='Reviewed inherited domain and at least four literal edge oracles; no downloaded code execution.')
            j=v%len(inputs);call=fn+'('+', '.join(repr(a) for a in inputs[j])+')'
            add('python',group,pair(f'Given this Python code:\n{answer}\nWhat does {call} return? Return only the Python value.',repr(expected[j])),identity,contrast_side=1,rules=[dict(kind='exact',args=[repr(expected[j])])],trace_contract=dict(code=code,function=fn,args_literal=repr(inputs[j]),expected_literal=repr(expected[j])))

    # Existing topics retain semantic IDs, so historical reserved topics do not
    # re-enter train via this new run number. Intact reviewed wording only.
    for topic,question,answer,short in TEXT:
        for prompt,target in [(question,answer),(question+' Keep the answer brief.',short)]:
            add('concise','reviewed-019:knowledge-'+topic,pair(prompt,target),review_contract='Reviewed full or concise answer; no clipping.')
    # Social and clarification replies are genuinely varied and tied to the
    # stated input; not one universal response repeated in every history.
    intents=[
      ('Hi, are you available?', 'Yes. What would you like help with?'),
      ('Good afternoon, can I ask a question?', 'Of course. What is your question?'),
      ('Hello, I need a hand.', 'Hello! Tell me what you need help with.'),
      ('Thanks for making that clear.', "You're welcome!"),
      ('I appreciate the explanation.', 'Glad it helped.'),
      ('That solves the problem, thank you.', 'Happy to help.'),
      ('I need to go now. Goodbye.', 'Goodbye! Take care.'),
      ('Thanks, I am done for now.', "You're welcome. See you next time!"),
      ('Can we chat for a moment?', 'Sure. What is on your mind?'),
      ('Are you ready to talk?', 'Yes, I am ready. What would you like to discuss?'),
    ]
    for i,(topic,a,b,x,y,p,q) in enumerate(SCENARIOS):
        group='contrast-021:scene-'+topic
        u,r=intents[i%len(intents)]
        add('reviewed_chat',group,pair(u,r)+pair(f'I want help with a short note about {topic}.',f'Who is the note for, and what should it say?'),review_contract='No invented biography or specialist role; request only genuinely missing details.')
        add('reviewed_chat',group,pair(f'Tell me two ways you can help with my {topic} project.', 'I can help organize your notes and draft a short checklist.'),review_contract='Exactly two plausible text-assistance capabilities.')
        add('reviewed_chat',group,pair(f'Can you summarize my {topic} notes? I have not included them.', 'Please paste the notes you want summarized.'),review_contract='Missing text is requested; no invented summary.')
        add('reviewed_chat',group,pair(f'I have an error in a Python program for my {topic} project.', 'Please paste the relevant code, the full error message, and what you expected to happen.'),review_contract='Request debugging evidence instead of inventing a cause.')
        add('reviewed_chat',group,pair(f'We are discussing {topic}. Thanks for your help.', ['You\'re welcome!', 'Glad I could help.', 'Happy to help.'][i%3]),review_contract='Contextual thanks, with no unsolicited restart.')
        add('reviewed_chat',group,pair(f'I need to leave our discussion about {topic}. Bye.', ['Goodbye! Take care.', 'See you next time.', 'Bye for now.'][i%3]),review_contract='Contextual farewell with no invented personal experience.')
    return rows
