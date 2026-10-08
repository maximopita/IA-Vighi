// Asocia cada etiqueta con el campo que tiene debajo (lectores de pantalla y clic en la etiqueta)
document.querySelectorAll("label:not([for])").forEach((lab, i) => {
  if (lab.querySelector("input, select, textarea")) return;
  const campo = lab.nextElementSibling;
  if (!campo || !/^(INPUT|SELECT|TEXTAREA)$/.test(campo.tagName)) return;
  campo.id = campo.id || `campo_${i}`;
  lab.htmlFor = campo.id;
});

// Cascada Subcategoría -> Sitio -> Tipo de muestra (formulario de ingreso)
(async function cascada() {
  const f = document.querySelector("form[data-tipo]");
  if (!f) return;
  const [sub, sitio, tm] = ["subcategoria", "sitio", "tipo_muestra"].map(n => f.elements[n]);
  const arbol = await (await fetch(`/api/catalogo/${f.dataset.tipo}`)).json();
  const llenar = (sel, opciones, actual) => {
    sel.innerHTML = '<option value=""></option>' + opciones.map(o =>
      `<option${o === actual ? " selected" : ""}>${o.replace(/</g, "&lt;")}</option>`).join("");
  };
  const actualizar = (desde) => {
    if (sub.tagName === "SELECT" && desde === "inicio") llenar(sub, Object.keys(arbol), sub.dataset.actual);
    const sitios = arbol[sub.value] || {};
    if (sitio.tagName === "SELECT" && desde !== "sitio") llenar(sitio, Object.keys(sitios), desde === "inicio" ? sitio.dataset.actual : "");
    llenar(tm, sitios[sitio.value] || [], desde === "inicio" ? tm.dataset.actual : "");
  };
  sub.addEventListener("change", () => actualizar("sub"));
  sitio.addEventListener("change", () => actualizar("sitio"));
  actualizar("inicio");
})();

// Templates: al elegir el título, el texto aparece en el campo editable
document.querySelectorAll("input[data-templates]").forEach(inp => {
  const lista = document.getElementById(inp.getAttribute("list"));
  const destino = document.getElementById(inp.dataset.destino);
  const concl = inp.dataset.conclusion ? document.getElementById(inp.dataset.conclusion) : null;
  let ultimo = inp.value;
  inp.addEventListener("change", async () => {
    const op = [...lista.options].find(o => o.value === inp.value);
    if (!op || inp.value === ultimo) return;
    const t = await (await fetch(`/api/template/${op.dataset.id}`)).json();
    const ocupado = destino.value.trim() || (concl && concl.value.trim());
    if (ocupado && !confirm("Ya hay texto escrito. ¿Reemplazarlo por el del template?")) { inp.value = ultimo; return; }
    destino.value = t.texto || "";
    if (concl) concl.value = t.conclusion || "";
    ultimo = inp.value;
    destino.focus();
  });
});

// Confirmaciones
document.querySelectorAll("form[data-confirmar]").forEach(f =>
  f.addEventListener("submit", e => { if (!confirm(f.dataset.confirmar)) e.preventDefault(); }));
