---
layout: leetcode-entry
title: "2359. Find Closest Node to Given Two Nodes"
permalink: "/leetcode/problem/2025-05-30-2359-find-closest-node-to-given-two-nodes/"
leetcode_ui: true
entry_slug: "2025-05-30-2359-find-closest-node-to-given-two-nodes"
---

[2359. Find Closest Node to Given Two Nodes](https://leetcode.com/problems/find-closest-node-to-given-two-nodes/description) medium
[blog post](https://leetcode.com/problems/find-closest-node-to-given-two-nodes/solutions/6794920/kotlin-rust-by-samoylenkodmitry-uulj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30052025-2359-find-closest-node-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/s7Z0xPqq3Po)
![1.webp](/assets/leetcode_daily_images/6f60aa4a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1004

#### Problem TLDR

Closest common node #medium #bfs

#### Intuition

Walk from two nodes in a parallel BFS by using two queues.
First intersection is the answer.

```j

    // 0 1 2
    // 2 0 0

    // 1 -. 0 .-. 2    a = 2, b = 0

    // 0 1 2 3 4 5  6
    // 5 4 5 4 3 6 -1

    // 0 -. 5 -. 6      a = 0 b = 1
    // 2 -.^
    //
    // 1 -. 4 .-. 3

```

#### Approach

* start with parallel BFS
* replace queues with single variables
* replace visited sets with marker variables in the edges

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 4ms
    fun closestMeetingNode(e: IntArray, a: Int, b: Int): Int {
        var a = a; var b = b; var r = e.size
        while (a >= 0 || b >= 0) {
            if (a >= 0) if (e[a] == -2) r = a else a = e[a].also { e[a] = -3 }
            if (b >= 0) if (e[b] == -3) r = min(r, b) else b = e[b].also { e[b] = -2 }
            if (r < e.size) return r
        }
        return -1
    }

```
```kotlin

// 54ms
    fun closestMeetingNode(e: IntArray, a: Int, b: Int): Int {
        val qa = ArrayDeque<Int>(); qa += a; var res = e.size
        val qb = ArrayDeque<Int>(); qb += b
        val va = HashSet<Int>(); val vb = HashSet<Int>();
        while (qa.size > 0 || qb.size > 0) {
            for (i in 0..<qa.size) {
                val x = qa.removeFirst()
                if (x in vb) res = min(res, x)
                if (va.add(x) && e[x] >= 0) qa += e[x]
            }
            for (i in 0..<qb.size) {
                val x = qb.removeFirst()
                if (x in va) res = min(res, x)
                if (vb.add(x) && e[x] >= 0) qb += e[x]
            }
            if (res < e.size) return res
        }
        return -1
    }

```
```rust

// 0ms
    pub fn closest_meeting_node(mut e: Vec<i32>, mut a: i32, mut b: i32) -> i32 {
        while a >= 0 || b >= 0 { let (i, j) = (a as usize, b as usize);
            if a >= 0 && e[i] == -2 { return if b >= 0 && e[j] == -3 { a.min(b) } else { a }}
            if a >= 0 { let x = e[i]; e[i] = -3; a = x }
            if b >= 0 { if e[j] == -3 { return b } else { let x = e[j]; e[j] = -2; b = x }}
        } -1
    }

```
```c++

// 0ms
    int closestMeetingNode(vector<int>& e, int a, int b) {
        while (a >= 0 || b >= 0) {
            if (a >= 0 && e[a] == -2) return b >= 0 && e[b] == -3 ? min(a, b) : a;
            if (a >= 0) { int t = e[a]; e[a] = -3; a = t; }
            if (b >= 0) if (e[b] == -3) return b; else { int t = e[b]; e[b] = -2, b = t; }
        } return -1;
    }

```

