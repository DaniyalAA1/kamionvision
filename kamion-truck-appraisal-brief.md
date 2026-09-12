# Kamion Challenge: What's This Truck Worth?

**Sponsor:** [Kamion](https://www.linkedin.com/company/kamion/) — $1,000 cash prize

---

## Who we are

Kamion is a Y Combinator-backed company and Türkiye's largest freight platform. Our app is a loadboard where truck drivers find cargo, buy fuel, and — most recently — buy and sell second-hand trucks.

That last part is the hard one. A used truck is a serious purchase, often the largest a small carrier will ever make. Buyers are looking at photos on a phone, hundreds of kilometers away, trying to work out whether a truck is worth what the seller is asking. Sellers, for their part, mostly have no idea what their truck is worth and price it by guessing. Listings sit for months. Deals fall apart over things that would have been obvious in person.

## The challenge

**Build something that looks at photos of a used truck and tells you what it's worth — and what's wrong with it.**
### Input

Photos of a truck. Assume a seller with a phone and no photography skills: bad lighting, awkward angles, mud, missing views. If you want to accept typed details as well (year, make, kilometers), go ahead — but the photos should do most of the work, because sellers leave things out and sometimes lie.
### Output

At minimum:

- **A price estimate**, with a range rather than a single number
- **A condition assessment** — what does the system actually see? Tire wear, rust, body damage, cab interior, anything a buyer would want flagged before driving six hours to look at it
- **Reasoning you can check (optional).** Why this price? What is it comparing against?

## What will win

A system that confidently prices a photo of a motorcycle, or a blurry night shot of nothing, is worse than useless in a real marketplace. Letting your system say "I can't tell from these photos, send me a shot of the tires" is a feature, not a cop-out.

Beyond that: does it survive contact with photos it has never seen? Is the reasoning specific enough that a buyer would trust it? Does it work?

**What won't win:** a thin wrapper that sends photos to a vision API and prints whatever number comes back. We'll be able to tell.

## Data

**We are not providing a dataset — collect your own.** Public listing sites have thousands of used truck listings with photos and asking prices. Gathering a few hundred is an hour of work and it's yours to keep after the weekend.

Kamion founder Berkay will be at the event. Feel free to ask us any questions you may have. 

## Judging

Four minutes of demo, three minutes of questions.

**Bring a working demo.** We will hand you photos of trucks that you have never seen and ask your system to appraise them live. Build for that.

|                              |                                                                           |
| ---------------------------- | ------------------------------------------------------------------------- |
| **Does it work?**            | Live demo, on our photos, without falling over                            |
| **Is it sensible?**          | Are the prices and condition notes defensible to someone who knows trucks |
| **Does it know its limits?** | Handles bad input, unclear photos, things that aren't trucks              |
| **Is it interesting?**       | Did you do something more thoughtful than the obvious approach            |

## Submit

A link to your repo and a demo screen recording. Send them to hack@kamion.co. 

---

*Questions any time during the event — find Kamion's founder or the event's organizers. We got you. 
