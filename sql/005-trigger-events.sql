-- 2,688,919 rows
create table user_mart_mihkel.cda_observed_anamnesis_tw_events_202603
as
  select
    e.id,
    e.epi_id,
    e."table",
    e.field,
    e.text,
    array_agg(distinct tw.word) as words
  from
    user_mart_mihkel.cda_observed_anamnesis_events_202603
    as e
    inner join
      (
        select distinct
          word
        from
          user_mart_mihkel.cda_trigger_words
      )
      as tw
    on true
  where
    e.text ~* tw.word
  group by e.id,
    e.epi_id,
    e."table",
    e.field,
    e.text;

comment on table user_mart_mihkel.cda_observed_anamnesis_tw_events_202603 is 'High recall trigger word filtered codrug discontinuation events';

create unique
index
if not exists "cda_observed_anamnesis_tw_events_202603_id_idx"
on user_mart_mihkel.cda_observed_anamnesis_tw_events_202603
using btree
(
  id
);
