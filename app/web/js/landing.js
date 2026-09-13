(() => {
  const world = document.querySelector('.world');
  const speech = document.getElementById('kip-speech');
  const lines = ['Beep beep.', 'Ask for another photo.', 'I look at photographs.', 'Parked.'];
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
    speech.textContent = driving ? 'Driving.' : 'Parked.';
  });
  const findings = [
    { title: 'Rear cab panel', body: 'The rear cab panel is visible. Accident history stays unproven from this angle.', next: 'Ask for front and side views to finish the exterior set.' },
    { title: 'Fifth wheel', body: 'The fifth wheel is visible behind the cab. The locking mechanism and mechanical condition stay unproven from this surface view.', next: 'Get a close-up of the coupling and have its operation checked in person.' },
    { title: 'Drive tires', body: 'The rear tires are visible. Tread depth and every sidewall stay unmeasured from this wide view.', next: 'Ask for close-ups of the tread and sidewalls on each tire.' }
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
