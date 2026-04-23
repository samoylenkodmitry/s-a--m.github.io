---
layout: leetcode-entry
title: "3408. Design Task Manager"
permalink: "/leetcode/problem/2025-09-18-3408-design-task-manager/"
leetcode_ui: true
entry_slug: "2025-09-18-3408-design-task-manager"
---

[3408. Design Task Manager](https://leetcode.com/problems/design-task-manager/description) medium
[blog post](https://leetcode.com/problems/design-task-manager/solutions/7201935/kotlin-rust-by-samoylenkodmitry-5a7v/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18092025-3408-design-task-manager?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HdtbT_rT1v0)

![1.webp](/assets/leetcode_daily_images/b33d4f9a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1116

#### Problem TLDR

Design Scheduler: add, edit, remove, execute by priority #medium #ds

#### Intuition

Use TreeMap + TreeSet

#### Approach

* we can use a single key mask for priority `p * 10^5 + tid`
* lazy removal seems to speed up; didn't implemented it here

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 358ms
class TaskManager(tasks: List<List<Int>>): TreeMap<Long, Int>() {
    val tp = HashMap<Int, Int>()
    init { for ((uid, tid, p) in tasks) add(uid, tid, p) }
    fun add(uid: Int, tid: Int, p: Int) { put(key(p, tid), uid); tp[tid] = p }
    fun key(p: Int, tid: Int): Long = 1L * p * 100_000 + tid
    fun key(tid: Int): Long = key(tp[tid]!!, tid)
    fun edit(tid: Int, p: Int) = add(rmv(tid), tid, p)
    fun rmv(tid: Int) = remove(key(tid)) ?: -1
    fun execTop(): Int = pollLastEntry()?.value ?: -1
}

```
```rust

// 118ms
#[derive(Default)] struct TaskManager(BTreeMap<(i32, i32), i32>, HashMap<i32, i32>);
impl TaskManager {
    fn new(v: Vec<Vec<i32>>) -> Self {
        let mut m = Self::default();
        for a in v { m.add(a[0], a[1], a[2]) }; m
    }
    fn add(&mut self, u: i32, i: i32, p: i32)
        { self.0.insert((p, i), u); self.1.insert(i, p); }
    fn edit(&mut self, i: i32, p: i32) { let u = self.rmv(i); self.add(u, i, p) }
    fn rmv(&mut self, i: i32) -> i32 {
        let k = (self.1[&i], i); self.1.remove(&i); self.0.remove(&k).unwrap_or(-1)
    }
    fn exec_top(&mut self) -> i32 { self.0.pop_last().map(|(_, v)| v).unwrap_or(-1) }
}

```

