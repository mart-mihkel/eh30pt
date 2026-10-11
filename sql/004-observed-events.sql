-- 5,164,851 rows
create table user_mart_mihkel.cda_observed_anamnesis_events_202603
as
  select
    de.id,
    de.epi_id,
    de."table",
    de.field,
    de.text
  from
    (
      select distinct on (d.patient_id, e.data ->> 'text')
        e.id,
        e.epi_id,
        e."table",
        e.field,
        e.data ->> 'text' as text,
        d.patient_id,
        d.episode
      from
        cda_fact_extraction_202603.events as e
        inner join
          user_sven.ddi194_observed_cda_documents as d
        using (
          "epi_id",
          "epi_type")
      where
        e."table" = 'anamnesis'
      order by e.data ->> 'text'
    )
    as de
  order by patient_id,
    episode,
    epi_id,
    id;

comment on table user_mart_mihkel.cda_observed_anamnesis_events_202603 is 'Distinct text events from user_sven.ddi194_observed_cda_documents';

create unique
index
if not exists "cda_observed_anamnesis_events_202603_id_idx"
on user_mart_mihkel.cda_observed_anamnesis_events_202603
using btree
(
  id
);
