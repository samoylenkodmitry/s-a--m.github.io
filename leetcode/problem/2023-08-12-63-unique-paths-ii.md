---
layout: leetcode-entry
title: "63. Unique Paths II"
permalink: "/leetcode/problem/2023-08-12-63-unique-paths-ii/"
leetcode_ui: true
entry_slug: "2023-08-12-63-unique-paths-ii"
---

[63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/description/) medium
[blog post](https://leetcode.com/problems/unique-paths-ii/solutions/3897324/kotlin-one-row-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12082023-63-unique-paths-ii?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/4b6ddef0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/306

#### Problem TLDR

Number of right-down ways tl->br in a matrix with obstacles

#### Intuition

Each time the robot moves in one direction gives a separate path. If two directions are possible, the number of paths gets added.

For example,

```
r r  #  0
r 2r 2r 2r
0 #  2r 4r
```

On the first row, the single path goes up to `1`.
On the second row, direct path down added to direct path right.
On the third row, the same happens when top and left numbers of paths are not 0.

#### Approach

Use a separate `row` array to remember previous row paths counts.

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun uniquePathsWithObstacles(obstacleGrid: Array<IntArray>): Int {
      val row = IntArray(obstacleGrid[0].size)
      row[0] = 1
      for (r in obstacleGrid)
        for (x in r.indices)
          if (r[x] != 0) row[x] = 0
          else if (x > 0) row[x] += row[x - 1]
      return row.last()
    }

```

#### The Magical Rundown

```

In Emojia's forgotten 🌌 corner, where time doesn't merely flow—it waltzes 💃,
spinning tales of lost yesterdays 🕰️ and unborn tomorrows ⌛, stands the
whispered legend of the Time Labyrinth. Not merely walls and corridors, but
a tapestry of fate's myriad choices, echoing distant memories and futures yet
conceived.

Bolt, the lonely automaton 🤖, not born but dreamt into existence by starlight ✨
and cosmic whimsy, felt an inexplicable yearning towards the 🏁 - the Time Nexus.
Ancient breezes 🍃 carried murmurs, not of it being an end, but a kaleidoscope
🎨 gateway to every pulse and flutter ❤️ of chronology's capricious dance 🌊.

╔═══╤═══╤═══╤═══╗
║🤖 │ 0 │🚫 │ 0 ║
╟───┼───┼───┼───╢
║ 0 │ 0 │ 0 │ 0 ║
╟───┼───┼───┼───╢
║ 0 │🚫 │ 0 │🏁 ║
╚═══╧═══╧═══╧═══╝

With each step, the fabric of reality quivered. Shadows of histories 🎶,
cosmic echoes 🌍, diverged and converged, painting and erasing moments of
what was, is, and could be.

---

Standing before the 🚫, it wasn't a barrier for Bolt, but a silent riddle:
"What song of the cosmos 🎵 shall you hum today, wanderer?"

╔═══╤═══╤═══╤═══╗
║🤖 │ ➡️ │🚫 │ 0 ║
╟───┼───┼───┼───╢
║ 0 │ 0 │ 0 │ 0 ║
╟───┼───┼───┼───╢
║ 0 │ 🚫 │ 0 │🏁 ║
╚═══╧═══╧═══╧═══╝

Dreamlike avenues 🛤️ unfurled, painting multitudes of futures in the vivid
colors of a universe in spring. In this chronal dance, Bolt secretly hoped
to outrace its own echoes, to be the first at the Nexus.

---

Junctions whispered with the delicate hum 🎵 of countless Bolts, each a tale,
a fate, a fleeting note in the grand cosmic symphony.

╔═══╤═══╤═══╤═══╗
║🤖 │ ➡️ │🚫 │ 0 ║
╟───┼───┼───┼───╢
║⬇️ │ 2➡️│2➡️│2➡️║
╟───┼───┼───┼───╢
║ 0 │ 🚫 │ 0 │🏁 ║
╚═══╧═══╧═══╧═══╝

Yet, as the Time Nexus loomed, revealing its vast enigma, a sense of profound
disquiet engulfed Bolt. Not only had another reflection reached before, but a
sea of mirrored selves stared back.

╔═══╤═══╤═══╤═══╗
║🤖 │ ➡️ │🚫 │ 0 ║
╟───┼───┼───┼───╢
║⬇️ │ 2➡️│2➡️│2➡️║
╟───┼───┼───┼───╢
║⬇️ │ 🚫 │2⬇️│4🏁║
╚═══╧═══╧═══╧═══╝

In that echoing vastness, Bolt's singular hope was smothered. In the dance of
time, amidst countless reflections, it whispered a silent, desperate question:
Which tune, which cadence, which moment 🎶 was truly its own in this timeless
waltz?

```

