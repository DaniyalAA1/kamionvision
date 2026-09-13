(() => {
  const world = document.querySelector('.world');
  const speech = document.getElementById('kip-speech');
  const lines = ['Beep beep. Due diligence coming through.', 'I like big trucks and I cannot lie.', 'My superpower? Asking for another photo.', '8 bits. Zero hidden agendas.', 'Keep your eyes on the evidence.'];
  let hello = 0;
  let hopTimer;
  document.getElementById('kip').addEventListener('click', () => {
    speech.textContent = lines[hello++ % lines.length];
    world.classList.remove('honk');
    void world.offsetWidth;
    world.classList.add('honk');
    clearTimeout(hopTimer);
    hopTimer = setTimeout(() => world.classList.remove('honk'), 550);
  });
  document.getElementById('cruise').addEventListener('click', event => {
    const driving = world.classList.toggle('cruising');
    event.currentTarget.setAttribute('aria-pressed', String(driving));
    event.currentTarget.textContent = driving ? 'PULL OVER ∥' : 'GO FOR A SPIN →';
    speech.textContent = driving ? 'Next stop: a better-informed decision.' : 'Parked. Ready for a closer look?';
  });
  const findings = [
    { title: 'The backstory<br>starts at the cab.', body: "The rear cab panel is visible. This angle helps locate exterior wear, but it can't establish accident history.", next: 'Ask for front and side views to complete the exterior picture.' },
    { title: 'The connection<br>is in the details.', body: 'The fifth wheel is visible behind the cab. A photo can show its surface, but not verify the locking mechanism or mechanical condition.', next: 'Get a close-up of the coupling and have its operation checked in person.' },
    { title: 'Tread carefully.<br>Literally.', body: "The rear tires are visible, but this wide view isn't enough to measure tread depth or assess every sidewall.", next: 'Ask for close-ups of the tread and sidewalls on each tire.' }
  ];
  const explored = new Set([0]);
  const buttons = document.querySelectorAll('[data-inspect]');
  buttons.forEach(button => button.addEventListener('click', () => {
    const index = Number(button.dataset.inspect);
    const finding = findings[index];
    explored.add(index);
    buttons.forEach(item => {
      const active = Number(item.dataset.inspect) === index;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    const copy = document.getElementById('inspection-copy');
    copy.querySelector('h3').innerHTML = finding.title;
    copy.querySelector(':scope > p').textContent = finding.body;
    copy.querySelector('.evidence-note p').textContent = finding.next;
    document.getElementById('inspected-count').textContent = `${explored.size} OF 3 AREAS EXPLORED${explored.size === 3 ? ' · GOOD EYE!' : ''}`;
    document.getElementById('explore-fill').style.width = `${explored.size / 3 * 100}%`;
  }));
})();
