---
layout: leetcode-entry
title: "1552. Magnetic Force Between Two Balls"
permalink: "/leetcode/problem/2024-06-20-1552-magnetic-force-between-two-balls/"
leetcode_ui: true
entry_slug: "2024-06-20-1552-magnetic-force-between-two-balls"
---

[1552. Magnetic Force Between Two Balls](https://leetcode.com/problems/magnetic-force-between-two-balls/description/) medium
[blog post](https://leetcode.com/problems/magnetic-force-between-two-balls/solutions/5339552/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20062024-1552-magnetic-force-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6UmtL0q_-Yw)
![2024-06-20_06-26_1.webp](/assets/leetcode_daily_images/235708cb.webp)

#### Join me no Telegram

https://t.me/leetcode_daily_unstoppable/645

#### Problem TLDR

Max shortest distance between `m` positions #medium #binary_search

#### Intuition

In a space of growing `shortest distance` we move from `impossible` to `possible` place `m` positions. Is Binary Search possible?

Let's try in example to check in a single pass `count` how many buckets we could place with given `shortest distance = 2`:

```j
    // 1 2 3 4 5 6 7 8    m=4
    // * *   * * * * *
    //   ^   ^   ^   ^
    // ^     ^   ^   ^
```

As we can see, two ways of placing possible, but there is no difference between choosing position `1` or `2`, so we can take positions `greedily`.

#### Approach

* we can skip using a separate variable for `max`, but in the interview it is better to use explicitly

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxDistance(position: IntArray, m: Int): Int {
        position.sort()
        var lo = 0; var hi = Int.MAX_VALUE
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            var count = 0; var next = 1
            for (p in position)
                if (p >= next) { count++; next = p + mid }
            if (count >= m) lo = mid + 1 else hi = mid - 1
        }
        return hi
    }

```
```rust

    pub fn max_distance(mut position: Vec<i32>, m: i32) -> i32 {
        position.sort_unstable(); let (mut lo, mut hi) = (0, i32::MAX);
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let (mut count, mut next) = (0, 1);
            for &p in &position { if p >= next { count += 1; next = p + mid }}
            if count >= m { lo = mid + 1 } else { hi = mid - 1 }
        }; hi
    }

```

