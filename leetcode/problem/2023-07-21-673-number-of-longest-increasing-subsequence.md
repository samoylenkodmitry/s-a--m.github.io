---
layout: leetcode-entry
title: "673. Number of Longest Increasing Subsequence"
permalink: "/leetcode/problem/2023-07-21-673-number-of-longest-increasing-subsequence/"
leetcode_ui: true
entry_slug: "2023-07-21-673-number-of-longest-increasing-subsequence"
---

[673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/) medium
[blog post](https://leetcode.com/problems/number-of-longest-increasing-subsequence/solutions/3795250/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/21072023-673-number-of-longest-increasing?sd=pf)
![image.png](/assets/leetcode_daily_images/e8b678c1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/282

#### Proble TLDR

Count of LIS in an array

#### Intuition

To find Longest Increasing Subsequence, there is a known algorithm with $$O(nlog(n))$$ time complexity. However, it can help with this case:

```bash

3 5 4 7

```

when we must track both `3 4 7` and `3 5 7` sequences. Given that, we can try to do full search with DFS, taking or skipping a number. To cache some results, we must make `dfs` depend on only the input arguments. Let's define it to return both `max length of LIS` and `count of them` in one result, and arguments are the starting position in an array and `previous number` that we must start sequence from.

#### Approach

* use an array cache, as `Map` gives TLE

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```

    class R(val maxLen: Int, val cnt: Int)
    fun findNumberOfLIS(nums: IntArray): Int {
      val cache = Array(nums.size + 1) { Array<R>(nums.size + 2) { R(0, 0) } }
      fun dfs(pos: Int, prevPos: Int): R = if (pos == nums.size) R(0, 1) else
        cache[pos][prevPos].takeIf { it.cnt != 0 }?: {
          val prev = if (prevPos == nums.size) Int.MIN_VALUE else nums[prevPos]
          var cnt = 0
          while (pos + cnt < nums.size && nums[pos + cnt] == nums[pos]) cnt++
          val skip = dfs(pos + cnt, prevPos)
          if (nums[pos] <= prev) skip else {
            val start = dfs(pos + cnt, pos).let { R(1 + it.maxLen, cnt * it.cnt ) }
            if (skip.maxLen == start.maxLen) R(skip.maxLen, start.cnt + skip.cnt)
            else if (skip.maxLen > start.maxLen) skip else start
          }
        }().also { cache[pos][prevPos] = it }
      return dfs(0, nums.size).cnt
    }

```

#### Magical rundown

```
🏰🔮🌌 The Astral Enigma of Eternity
In the boundless tapestry of time, an enigmatic labyrinth 🗝️ whispers
tales of forgotten epochs. Your fateful quest? To decipher the longest
increasing subsequences hidden within the celestial array 🧩 [3, 5, 4, 7].

🌄 The Aurora Gateway: dfs(0, nums.size)
    /                          \
🌳 The Verdant Passage (dfs(1,0)) / 🌑 The Nebulous Veil (dfs(1,nums.size))

Your odyssey commences at twilight's brink: will you tread the lush
🌳 Verdant Passage or dare to penetrate the enigmatic 🌑 Nebulous Veil?

🌄 The Aurora Gateway: dfs(0, nums.size)
   /
🍃 The Glade of Whispers (Pos 1: num[1]=3, dfs(1,0))
   /
🌊 The Cascade of Echoes (Pos 2: num[2]=5, dfs(2,1))
   /
⛰️ The Bastion of Silence (Pos 3: num[3]=4, dfs(3,2)) 🚫🔒

The labyrinth’s heart pulsates with cryptic riddles. The ⛰️ Bastion of Silence
remains locked, overshadowed by the formidable 🌊 Cascade of Echoes.

🌄 The Aurora Gateway: dfs(0, nums.size)
   /
🍃 The Glade of Whispers (Pos 1: num[1]=3, dfs(1,0))
   \
🌑 The Phantom of Riddles (Pos 2: num[2]=5, dfs(2,0))

Retracing your footsteps, echoes of untaken paths whisper secrets. Could
the ⛰️ Bastion of Silence hide beneath the enigma of the 🌑 Phantom of Riddles?

🌄 The Aurora Gateway: dfs(0, nums.size)
   /
🍃 The Glade of Whispers (Pos 1: num[1]=3, dfs(1,0))
   \
💨 The Mist of Mystery (Pos 3: num[3]=4, dfs(3,0))
   \
🌩️ The Tempest of Triumph (Pos 4: num[4]=7, dfs(4,3)) 🏁🎉

At last, the tempest yields! Each twist and turn, each riddle spun and
secret learned, illuminates a longest increasing subsequence in the cosmic array.

Your enchanted grimoire 📜✨ (cache) now vibrates with the wisdom of ages:

prevPos\pos  0     1      2      3     4
       0     (0,0) (2,1) (2,1)  (3,2) (0,0)
       1     (0,0) (0,0) (2,1)  (3,2) (0,0)
       2     (0,0) (0,0) (0,0)  (2,1) (0,0)
       3     (0,0) (0,0) (0,0)  (0,0) (0,0)
       4     (0,0) (0,0) (0,0)  (0,0) (0,0)

Beneath the shimmering cosmic symphony, you cast the final incantation
🧙‍♂️ dfs(0, nums.size).cnt. The grimoire blazes with ethereal light, revealing
the total count of longest increasing subsequences.

You emerge from the labyrinth transformed: no longer merely an adventurer,
but the 🌟 Cosmic Guardian of Timeless Wisdom. 🗝️✨🌠

```

