create schema if not exists "user_mart_mihkel";

-- 2,180 rows
create table if not exists user_mart_mihkel.cda_trigger_words (
  id bigserial primary key,
  word text not null,
  subtype text not null
);

comment on table user_mart_mihkel.cda_trigger_words is 'Trigger words for drug discontinuation';

insert into user_mart_mihkel.cda_trigger_words (word, subtype)
select stem || ending, 'verb'
from
  (values
    ('katkesta'),
    ('lõpeta'),
    ('asenda'),
    ('vaheta'),
    ('loobu'),
    ('vähenda'),
    ('tühista'),
    ('peata'),
    ('jäta'),
    ('arva'),
    ('soovi'),
    ('taha'),
    ('otsusta'),
    ('pane'),
    ('määra'),
    ('muuda'),
    ('eemalda'),
    ('kärbi'),
    ('piira'),
    ('lükka')
  )
  as verb_stem(stem)
  cross join
  (values
    (''),
    ('b'),
    ('d'),
    ('n'),
    ('s'),
    ('v'),
    ('da'),
    ('ge'),
    ('gu'),
    ('ks'),
    ('ma'),
    ('me'),
    ('ta'),
    ('te'),
    ('ti'),
    ('des'),
    ('gem'),
    ('mas'),
    ('nud'),
    ('sid'),
    ('sin'),
    ('tav'),
    ('tud'),
    ('vad'),
    ('vat'),
    ('ksid'),
    ('ksin'),
    ('maks'),
    ('mast'),
    ('mata'),
    ('nuks'),
    ('sime'),
    ('site'),
    ('tagu'),
    ('taks'),
    ('tama'),
    ('tuks'),
    ('ksime'),
    ('ksite'),
    ('nuvat'),
    ('takse'),
    ('tavat'),
    ('tuvat'),
    ('nuksid'),
    ('nuksin'),
    ('nuksime'),
    ('nuksite')
  )
  as verb_ending(ending)
union all
select stem || ending, 'noun'
from
  (values
    ('katkesta'),
    ('lõpeta'),
    ('asenda'),
    ('vaheta'),
    ('loobu')
  )
  as noun_stem(stem)
  cross join
  (values
    ('mine'),
    ('mise'),
    ('misi'),
    ('mist'),
    ('mised'),
    ('misel'),
    ('mises'),
    ('misil'),
    ('misis'),
    ('misse'),
    ('miste'),
    ('misega'),
    ('miseks'),
    ('misele'),
    ('miselt'),
    ('misena'),
    ('miseni'),
    ('misest'),
    ('miseta'),
    ('misiks'),
    ('misile'),
    ('misilt'),
    ('misist'),
    ('mistel'),
    ('mistes'),
    ('misesse'),
    ('misisse'),
    ('mistega'),
    ('misteks'),
    ('mistele'),
    ('mistelt'),
    ('mistena'),
    ('misteni'),
    ('mistest'),
    ('misteta'),
    ('mistesse')
  )
  as noun_ending(ending);
