create schema if not exists "user_mart_mihkel";

-- 2,180 rows
create table if not exists user_mart_mihkel.cda_trigger_words (
  id bigserial primary key,
  word text not null,
  subtype text not null
);

comment on table user_mart_mihkel.cda_trigger_words is 'Trigger words for drug discontinuation';

insert into user_mart_mihkel.cda_trigger_words (word, subtype)
select
  stem || ending,
  'verb'
from
  (
    values
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
      ('lükka'),
      ('tõsta'),
      ('suurenda'),
      ('võta'),
      ('tarvita'),
      ('jätka'),
      ('alusta'),
      ('unusta'),
      ('kohanda'),
      ('taasta'),
      ('paranda')
  )
  as verb_stem (stem)
  inner join
    (
      values
        (''),
        ('b'),
        ('d'),
        ('n'),
        ('s'),
        ('v'),
        ('da'),
        ('ma'),
        ('des'),
        ('me'),
        ('te'),
        ('ti'),
        ('tud'),
        ('nud'),
        ('vad'),
        ('vat'),
        ('gu'),
        ('ge'),
        ('ks'),
        ('taks')
    )
    as verb_ending (ending)
  on true
union all
select
  stem || ending,
  'noun'
from
  (
    values
      ('katkesta'),
      ('lõpeta'),
      ('asenda'),
      ('vaheta'),
      ('loobu'),
      ('tarvita'),
      ('vähenda'),
      ('peata'),
      ('tühista'),
      ('eemalda')
  )
  as noun_stem (stem)
  inner join
    (
      values
        ('mine'),
        ('mise'),
        ('misi'),
        ('mist'),
        ('mised'),
        ('misel'),
        ('mises'),
        ('misega'),
        ('miseks'),
        ('misele'),
        ('miselt'),
        ('misena'),
        ('miseni'),
        ('misest'),
        ('mistel'),
        ('mistes')
    )
    as noun_ending (ending)
  on true;
