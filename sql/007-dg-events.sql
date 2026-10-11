-- 18,187,184 rows
create table user_mart_mihkel.dg_cda_observed_anamnesis_events_202603
as
  select
    de.id,
    de.epi_id,
    de."table",
    de.field,
    de.text,
    de.event_type
  from
    (
      select distinct on (
        d.patient_id,
        e.data ->> 'text',
        d.event_type
      )
        e.id,
        e.epi_id,
        e."table",
        e.field,
        e.data ->> 'text' as text,
        d.patient_id,
        d.episode,
        d.event_type
      from
        cda_fact_extraction_202603.events as e
        inner join
          user_sven.dg_observed_cda_documents as d
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

comment on table user_mart_mihkel.dg_cda_observed_anamnesis_events_202603 is 'Distinct text events from user_sven.dg_observed_cda_documents';

create unique
index
if not exists "dg_cda_observed_anamnesis_events_202603_id_idx"
on user_mart_mihkel.dg_cda_observed_anamnesis_events_202603
using btree
(
  id
);
