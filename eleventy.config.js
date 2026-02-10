module.exports = function(eleventyConfig) {
  // Copy assets folder
  eleventyConfig.addPassthroughCopy("src/assets");

  // Date formatting filter
  eleventyConfig.addFilter("dateFormat", (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  });

  // Short date filter
  eleventyConfig.addFilter("shortDate", (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  });

  // Group meetings by committee
  eleventyConfig.addFilter("groupByCommittee", (meetings) => {
    const groups = {};
    for (const meeting of meetings) {
      const committee = meeting.committee || 'Uncategorized';
      if (!groups[committee]) {
        groups[committee] = [];
      }
      groups[committee].push(meeting);
    }
    return groups;
  });

  // Sort by date descending
  eleventyConfig.addFilter("sortByDateDesc", (meetings) => {
    return [...meetings].sort((a, b) => new Date(b.date) - new Date(a.date));
  });

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data"
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk"
  };
};
