---
layout: leetcode-entry
title: "380. Insert Delete GetRandom O(1)"
permalink: "/leetcode/problem/2024-01-16-380-insert-delete-getrandom-o-1/"
leetcode_ui: true
entry_slug: "2024-01-16-380-insert-delete-getrandom-o-1"
---

[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/description/) medium
[blog post](https://leetcode.com/problems/insert-delete-getrandom-o1/solutions/4573497/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16012024-380-insert-delete-getrandom?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/MTxB74kat0k)
![image.png](/assets/leetcode_daily_images/c74a3e60.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/473

#### Problem TLDR

Implement HashSet<Int> with random method.

#### Intuition

There is a `random` method exists in Kotlin's `MutableSet` https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/random.html.

However, let's just use array to store values and save positions in a `HashMap`. The order in array didn't matter, so we can remove elements in O(1).

#### Approach

To save some symbols of code, we can extend from ArrayList.

#### Complexity

- Time complexity:
$$O(1)$$, per operation

- Space complexity:
$$O(1)$$, per operation

#### Code

```kotlin

class RandomizedSet(): ArrayList<Int>() {
  val vToPos = HashMap<Int, Int>()
  fun insert(v: Int): Boolean {
    if (vToPos.contains(v)) return false
    add(v)
    vToPos[v] = lastIndex
    return true
  }
  override fun remove(v: Int): Boolean {
    val pos = vToPos.remove(v) ?: return false
    set(pos, last())
    if (last() != v) vToPos[last()] = pos
    removeLast()
    return true
  }
  fun getRandom() = random()
}

```

```rust

use rand::{thread_rng, Rng};
use std::collections::HashMap;

struct RandomizedSet {
  vec: Vec<i32>,
  v_to_i: HashMap<i32, usize>,
}

impl RandomizedSet {

  fn new() -> Self {
    Self { vec: vec![], v_to_i: HashMap::new() }
  }

  fn insert(&mut self, v: i32) -> bool {
    if self.v_to_i.entry(v).or_insert(self.vec.len()) != &self.vec.len() {
      return false;
    }
    self.vec.push(v);
    true
  }

  fn remove(&mut self, v: i32) -> bool {
    self.v_to_i.remove(&v).map_or(false, |i| {
      let last = self.vec.pop().unwrap();
      if (last != v) {
        self.vec[i] = last;
        self.v_to_i.insert(last, i);
      }
      true
    })
  }

  fn get_random(&self) -> i32 {
    self.vec[thread_rng().gen_range(0, self.vec.len())]
  }
}

```

