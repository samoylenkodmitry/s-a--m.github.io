---
layout: leetcode-entry
title: "1642. Furthest Building You Can Reach"
permalink: "/leetcode/problem/2024-02-17-1642-furthest-building-you-can-reach/"
leetcode_ui: true
entry_slug: "2024-02-17-1642-furthest-building-you-can-reach"
---

[1642. Furthest Building You Can Reach](https://leetcode.com/problems/furthest-building-you-can-reach/description) medium
[blog post](https://leetcode.com/problems/furthest-building-you-can-reach/solutions/4740195/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17022024-1642-furthest-building-you?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8BFQk0vFB78)
![image.png](/assets/leetcode_daily_images/397799a6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/509

#### Problem TLDR

Max index to climb diff = a[i +1] - a[i] > 0 using bricks -= diff and ladders-- for each.

#### Intuition

First, understand the problem by observing the inputs:

```bash

  // 0 1  2 3 4 5  6  7 8
  // 4 12 2 7 3 18 20 3 19    10 2
  //  8    5   15 2    16
  //  b    l   l  b

```
* only increasing pairs matters
* it is better to use the `ladders` for the biggest `diffs`

The simple solution without tricks is to do a BinarySearch: can we reach the `mid`-point using all the bricks and ladders? Then just sort diffs in `0..mid` range and take `brick`s for the smaller and `ladders` for the others. This solution would cost us O(nlog^2(n)) and it passes.

However, in the leetcode comments, I spot that there is an O(nlogn) solution exists. The idea is to grab as much bricks as we can and if we cannot, then we can *drop back* some (biggest) pile of bricks and *pretend* we used the ladders instead. We can do this trick at most `ladders`' times.

#### Approach

Try not to write the `if` checks that are irrelevant.
* BinaryHeap in Rust is a `max` heap
* PriorityQueue in Kotlin is a `min` heap, use `reverseOrder`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun furthestBuilding(heights: IntArray, bricks: Int, ladders: Int): Int {
    val pq = PriorityQueue<Int>(reverseOrder())
    var b = bricks; var l = ladders
    for (i in 1..<heights.size) {
      val diff = heights[i] - heights[i - 1]
      if (diff <= 0) continue
      pq += diff
      if (b < diff && l-- > 0) b += pq.poll()
      if (b < diff) return i - 1
      b -= diff
    }
    return heights.lastIndex
  }

```
```rust

  pub fn furthest_building(heights: Vec<i32>, mut bricks: i32, mut ladders: i32) -> i32 {
    let mut hp = BinaryHeap::new();
    for i in 1..heights.len() {
      let diff = heights[i] - heights[i - 1];
      if diff <= 0 { continue }
      hp.push(diff);
      if bricks < diff && ladders > 0 {
        bricks += hp.pop().unwrap();
        ladders -= 1;
      }
      if bricks < diff { return i as i32 - 1 }
      bricks -= diff;
    }
    heights.len() as i32 - 1
  }

```

