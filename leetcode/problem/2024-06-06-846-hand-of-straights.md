---
layout: leetcode-entry
title: "846. Hand of Straights"
permalink: "/leetcode/problem/2024-06-06-846-hand-of-straights/"
leetcode_ui: true
entry_slug: "2024-06-06-846-hand-of-straights"
---

[846. Hand of Straights](https://leetcode.com/problems/hand-of-straights/description/) medium
[blog post](https://leetcode.com/problems/hand-of-straights/solutions/5266860/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06062024-846-hand-of-straights?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ds03wmXeDd0)
![2024-06-06_07-37_1.webp](/assets/leetcode_daily_images/c998db04.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/631

#### Problem TLDR

Can array be split into consecutive groups #medium #heap #treemap

#### Intuition

Let's sort array and try to brute-force solve it with bare hands:

```j
    // 1 2 3 6 2 3 4 7 8

    // 0 1 2 3 4 5 6 7 8
    // 1 2 2 3 3 4 6 7 8
    // 1 2   3
    //     2   3 4
    //             6 7 8

    // 1 2 3 4 5 6       2

    // 1 2 3             1
```
The naive implementation is accepted: take first not used and mark all consequtive until `groupSize` reached. This solution will take O(n^2) time, but it is fast as arrays are fast when iterated forward.

To improve we can use PriorityQueue: do the same algorithm, skip the duplicated, then add them back. This will take O(nlogn + gk), where g is groups count, and k is duplicates count.

We can improve event more with TreeMap: keys are the hands, values are the counters, subtract entire `count`.

#### Approach

Let's implement both PriorityQueue and TreeMap solutions.

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun isNStraightHand(hand: IntArray, groupSize: Int): Boolean {
        val map = TreeMap<Int, Int>()
        for (h in hand) map[h] = 1 + (map[h] ?: 0)
        for ((h, count) in map) if (count > 0)
            for (x in h..<h + groupSize) {
                if ((map[x] ?: 0) < count) return false
                map[x] = map[x]!! - count
            }
        return true
    }

```
```rust

    pub fn is_n_straight_hand(hand: Vec<i32>, group_size: i32) -> bool {
        let mut bh = BinaryHeap::new(); for &h in &hand { bh.push(-h); }
        while let Some(start) = bh.pop() {
            let mut tmp = vec![];
            for i in -start + 1..-start + group_size {
                while bh.len() > 0 && -bh.peek().unwrap() < i { tmp.push(bh.pop().unwrap()); }
                if bh.is_empty() || -bh.peek().unwrap() > i { return false }
                bh.pop();
            }
            for &h in &tmp { bh.push(h); }
        }; true
    }

```

