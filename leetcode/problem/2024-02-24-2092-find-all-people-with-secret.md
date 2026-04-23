---
layout: leetcode-entry
title: "2092. Find All People With Secret"
permalink: "/leetcode/problem/2024-02-24-2092-find-all-people-with-secret/"
leetcode_ui: true
entry_slug: "2024-02-24-2092-find-all-people-with-secret"
---

[2092. Find All People With Secret](https://leetcode.com/problems/find-all-people-with-secret/description/) hard
[blog post](https://leetcode.com/problems/find-all-people-with-secret/solutions/4775018/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24022024-2092-find-all-people-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3a91b826JmI)

![image.png](/assets/leetcode_daily_images/b11cd1ad.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/517

#### Problem TLDR

Who knows 0 and firstPerson's secret after group meetings at times: [personA, personB, time].

#### Intuition

To share the secret between people we can use a known Union-Find data structure. The corner case is when the meeting time is passed and no one knowns a secret: we must revert a union for these people.

#### Approach

To make Union-Find more performant, there are several tricks. One of them is a path compression: after finding the root, set all the intermediates to root. Ranks are more complex and not worth the lines of code.

#### Complexity

- Time complexity:
$$O(an)$$, `a` is close to 1

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findAllPeople(n: Int, meetings: Array<IntArray>, firstPerson: Int): List<Int> {
    meetings.sortWith(compareBy { it[2] })
    val uf = HashMap<Int, Int>()
    fun root(a: Int): Int =
      uf[a]?.let { if (a == it) a else root(it).also { uf[a] = it } } ?: a
    uf[0] = firstPerson
    val s = mutableListOf<Int>()
    var prev = 0
    for ((a, b, t) in meetings) {
      if (t > prev) for (x in s) if (root(x) != root(0)) uf[x] = x
      if (t > prev) s.clear()
      uf[root(a)] = root(b)
      s += a; s += b; prev = t
    }
    return (0..<n).filter { root(0) == root(it) }
  }

```
```rust

  pub fn find_all_people(n: i32, mut meetings: Vec<Vec<i32>>, first_person: i32) -> Vec<i32> {
    meetings.sort_unstable_by_key(|m| m[2]);
    let mut uf: Vec<_> = (0..n as usize).collect();
    fn root(uf: &mut Vec<usize>, mut x: usize) -> usize {
      while uf[x] != x { uf[x] = uf[uf[x]]; x = uf[x] } x
    }
    uf[0] = first_person as _;
    let (mut prev, mut s) = (0, vec![]);
    for m in &meetings {
      if m[2] > prev { for &x in &s { if root(&mut uf, x) != root(&mut uf, 0) { uf[x] = x }}}
      if m[2] > prev { s.clear() }
      let ra = root(&mut uf, m[0] as _);
      uf[ra] = root(&mut uf, m[1] as _);
      s.push(m[0] as _); s.push(m[1] as _); prev = m[2]
    }
    (0..n).filter(|&x| root(&mut uf, x as _) == root(&mut uf, 0)).collect()
  }

```

