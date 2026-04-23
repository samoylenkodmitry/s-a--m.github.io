---
layout: leetcode-entry
title: "3508. Implement Router"
permalink: "/leetcode/problem/2025-09-20-3508-implement-router/"
leetcode_ui: true
entry_slug: "2025-09-20-3508-implement-router"
---

[3508. Implement Router](https://leetcode.com/problems/implement-router/description/) medium
[blog post](https://leetcode.com/problems/implement-router/solutions/7207927/kotlin-rust-by-samoylenkodmitry-78w0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20092025-3508-implement-router?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6tf7Rb_v9uc)

![1.webp](/assets/leetcode_daily_images/6577d623.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1118

#### Problem TLDR

Design Router: addPacket, forwardPacket, getCount(dst, ts start..end) #medium #ds

#### Intuition

The main difficulty is the `getCount`, we have to maintain some sorted order of timestamps, but there are duplicates.
* use map `dst to sorted timestamps` for getCount; do binarysearch
* use `LinkedList` or `ArrayDeque` or `IntArray(limit)` for FIFO adding/removal
* use `HashSet` to skip duplicates

#### Approach

* remember binary search: always check `lo <= hi`, always do `hi=m-1 or lo=m+1`, update value if in condition
* skip the second binary search if the first gives out of range idx
* when removing `forwardPacket` from `byDst` list we always remove the first (it is theoretically O(n) call, but there is no testcase for this; to improve have to track pointer to first and do garbage collection)

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 291ms
class Router(val limit: Int) : LinkedHashSet<List<Int>>() {
    val byDst = HashMap<Int, ArrayList<Int>>()
    fun addPacket(src: Int, dst: Int, ts: Int) =
        add(listOf(src,dst,ts)) && { if (size > limit) forwardPacket()
        byDst.getOrPut(dst) { ArrayList() } += ts; true }()
    fun forwardPacket() = firstOrNull()?.also {
            this -= it; byDst[it[1]]?.removeFirst()
        }?.toIntArray() ?: intArrayOf()
    fun getCount(dst: Int, st: Int, et: Int) = byDst[dst]?.run {
        binarySearch { if (it < st) -1 else 1 } -
        binarySearch { if (it <= et) -1 else 1 } } ?: 0
}

```
```rust

// 76ms
#[derive(Default)] struct Router(i32, VecDeque<[i32;3]>,HashMap<i32,Vec<i32>>,HashSet<[i32;3]>);
impl Router {
    fn new(l: i32) -> Self { let mut s = Self::default(); s.0 = l; s }
    fn add_packet(&mut self, s: i32, d: i32, t: i32) -> bool {
        let k = [s,d,t]; self.3.insert(k) && {
        self.1.push_back(k); self.2.entry(d).or_default().push(t);
        if self.1.len() as i32 > self.0 { self.forward_packet(); } true }
    }
    fn forward_packet(&mut self) -> Vec<i32> {
        self.1.pop_front().map(|k|{
            self.3.remove(&k); self.2.get_mut(&k[1]).map(|v|v.remove(0)); k.into()
        }).unwrap_or_default()
    }
    fn get_count(&self, d: i32, s: i32, e: i32) -> i32 {
        self.2.get(&d).map_or(0, |v|v.partition_point(|&x|x<=e)-v.partition_point(|&x|x<s)) as _
    }
}

```

